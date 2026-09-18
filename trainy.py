import argparse
import copy
import csv
from contextlib import nullcontext
from datetime import datetime
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Prefer this script's sibling model.py over unrelated installed packages.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import WMNet

# 所有配置集中在这里，无需额外配置文件。
MODEL_CONFIG = {'in_channels': 1,
 'out_channels': 1,
 'channels': (64, 128, 256, 512),
 'num_res_blocks': 6,
 'embedding_dim': 128,
 'embedding_hidden': 512,
 'norm_groups': 8,
 'noise_scale': 1000.0,
 'gradient_checkpointing': True}

TRAIN_CONFIG = {'epochs': 500,
 'batch_size': 4,
 'micro_batch_size': 1,
 'lr': 0.0002,
 'min_lr': 5e-05,
 'lr_power': 0.9,
 'betas': (0.5, 0.999),
 'weight_decay': 0.0,
 'l1_weight': 100.0,
 'amp': True,
 'seed': 42,
 'deterministic': False,
 'val_interval': 1,
 'checkpoint_interval': 25,
 'log_interval': 20,
 'selection_metric': 'psnr',
 'device': 'auto',
 'cpu_threads': 4}

DATA_CONFIG = {'train_noise': 'datasets/train/noise',
 'train_clean': 'datasets/train/clean',
 'val_noise': 'datasets/val/noise',
 'val_clean': 'datasets/val/clean',
 'patch_size': 512,
 'resize': None,
 'workers': 0,
 'intensity_max': None}

EVALUATION_CONFIG = {'crop_border': 0, 'clamp': True, 'ssim_window': 11, 'ssim_sigma': 1.5}

WORK_DIR = 'work_dirs/wmnet'
EXTENSIONS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_checkpoint(path):
    kwargs = {'map_location': 'cpu'}
    # Training checkpoints contain optimizer/RNG state; use only trusted files.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def atomic_save(state, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, str(tmp))
    os.replace(str(tmp), str(path))


def save_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    # Older cuDNN searches can allocate large transient workspaces.
    # Keep the original ZIP's benchmark=False setting.
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def polynomial_lr(step, total_steps, initial, minimum, power):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return minimum + (initial - minimum) * (1.0 - progress) ** power


def read_grayscale(path, intensity_max=None):
    with Image.open(path) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Multi-page images need explicit frame extraction: ' + str(path))
        if im.mode in ('RGB', 'RGBA', 'P', 'CMYK'):
            # Matches the old 8-bit grayscale workflow; alpha is not a channel.
            a = np.asarray(im.convert('L'))
        else:
            a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError('Expected a single B-scan: ' + str(path))
    if intensity_max is not None:
        denominator = float(intensity_max)
    elif a.dtype == np.uint8:
        denominator = 255.0
    elif a.dtype == np.uint16 or (a.dtype == np.int32 and a.min() >= 0 and a.max() <= 65535):
        denominator = 65535.0
    elif np.issubdtype(a.dtype, np.floating):
        denominator = 1.0
    else:
        raise ValueError('Unsupported image dtype; set data.intensity_max: ' + str(a.dtype))
    if denominator <= 0:
        raise ValueError('intensity_max must be positive')
    a = a.astype(np.float32) / denominator
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1.000001:
        raise ValueError('Image outside [0,1]; specify its acquisition intensity_max: ' + str(path))
    return np.ascontiguousarray(a)


class PairedDataset(Dataset):
    def __init__(self, rows, root, cfg, training=False):
        self.rows, self.root, self.cfg, self.training = rows, root, cfg, training
        self.epoch = 0
        self.seed = 42

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        noisy = read_grayscale(self.root / row['noise'], self.cfg.get('intensity_max'))
        clean = read_grayscale(self.root / row['clean'], self.cfg.get('intensity_max'))
        if noisy.shape != clean.shape:
            raise ValueError('Unaligned pair dimensions: ' + row['noise'])
        pair = torch.from_numpy(np.stack((noisy, clean)))[:, None]
        resize = self.cfg.get('resize')
        if resize:
            pair = F.interpolate(pair, size=(resize[1], resize[0]), mode='bilinear', align_corners=False)
        size = self.cfg['patch_size']
        height, width = pair.shape[-2:]
        if height < size or width < size:
            raise ValueError('Image smaller than patch_size; provide real 512x512 patches or set resize explicitly: ' + row['noise'])
        # Per-index/epoch RNG makes augmentation independent of worker scheduling
        # and reproducible across epoch-boundary checkpoint resume.
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        top = rng.randint(0, height - size) if self.training else (height - size) // 2
        left = rng.randint(0, width - size) if self.training else (width - size) // 2
        pair = pair[:, :, top:top+size, left:left+size]
        if self.training:
            if rng.random() < 0.5: pair = pair.flip(-1)
            if rng.random() < 0.5: pair = pair.flip(-2)
            if rng.random() < 0.5: pair = pair.transpose(-1, -2)
        return dict(lq=pair[0].contiguous(), gt=pair[1].contiguous(), name=row['noise'])


def update_best_metrics(previous, current, epoch):
    """Track each validation maximum with both scores from that same epoch.

    Do not combine a PSNR from one model with an SSIM from another model and
    present the pair as the result of a single checkpoint.
    """
    result = dict(previous)
    improved = []
    for metric in ('psnr', 'ssim'):
        if metric in current and (metric not in result or current[metric] > result[metric]['value']):
            result[metric] = dict(value=current[metric], epoch=epoch,
                                  psnr=current['psnr'], ssim=current['ssim'])
            improved.append(metric)
    return result, improved


def image_metrics(prediction, target, crop_border=0, clamp=True, ssim_window=11, ssim_sigma=1.5):
    x, y = prediction.float(), target.float()
    if clamp:
        x = x.clamp(0, 1)
    if crop_border:
        x = x[..., crop_border:-crop_border, crop_border:-crop_border]
        y = y[..., crop_border:-crop_border, crop_border:-crop_border]
    if min(x.shape[-2:]) < ssim_window or ssim_window % 2 != 1:
        raise ValueError('SSIM requires an odd window <= image dimensions after border crop')
    mse = (x - y).square().mean((1, 2, 3))
    psnr = -10 * torch.log10(mse)
    grid = torch.arange(ssim_window, device=x.device, dtype=torch.float32) - ssim_window // 2
    kernel = torch.exp(-grid.square() / (2 * ssim_sigma**2))
    kernel /= kernel.sum()
    window = (kernel[:, None] * kernel[None, :])[None, None]
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    var_x = (F.conv2d(x*x, window) - mu_x*mu_x).clamp_min(0)
    var_y = (F.conv2d(y*y, window) - mu_y*mu_y).clamp_min(0)
    cov = F.conv2d(x*y, window) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + 0.01**2) * (2*cov + 0.03**2) /
                ((mu_x.square()+mu_y.square()+0.01**2) * (var_x+var_y+0.03**2)))
    return psnr, ssim_map.mean((1, 2, 3))


def select_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Activate your GPU PyTorch environment or use --device cpu.')
    return device


@torch.no_grad()
def evaluate(model, loader, device, cfg, amp=False, preview_path=None, per_image=False):
    model.eval()
    records = []
    preview = None
    for batch in loader:
        x, y = batch['lq'].to(device), batch['gt'].to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(x)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Non-finite validation prediction')
        l1 = (prediction.float() - y).abs().mean((1, 2, 3))
        psnr, ssim = image_metrics(prediction, y, **cfg['evaluation'])
        raw_psnr, raw_ssim = image_metrics(x, y, **cfg['evaluation'])
        for i, name in enumerate(batch['name']):
            records.append(dict(name=name, l1=l1[i].item(),
                                loss=cfg['train']['l1_weight'] * l1[i].item(),
                                psnr=psnr[i].item(), ssim=ssim[i].item(),
                                raw_psnr=raw_psnr[i].item(), raw_ssim=raw_ssim[i].item()))
        if preview is None and preview_path is not None:
            preview = torch.cat((x[0, 0], prediction[0, 0].float().clamp(0, 1), y[0, 0]), dim=1).cpu().numpy()
    if not records:
        raise ValueError('Empty evaluation loader')
    metrics = {key: float(np.mean([r[key] for r in records])) for key in records[0] if key != 'name'}
    if preview is not None:
        Image.fromarray(np.round(preview * 255).astype(np.uint8)).save(preview_path)
    return (metrics, records) if per_image else metrics


def train(cfg, args):
    settings = cfg['train']
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))
    distributed = args.launcher == 'pytorch' and world_size > 1
    if world_size > 1 and not distributed:
        raise ValueError('Distributed launch requires --launcher pytorch')
    if args.launcher not in ('none', 'pytorch'):
        raise ValueError('WM-Net supports none/pytorch launchers; use none or pytorch')
    if args.gpus != 1 and not distributed:
        raise ValueError('For multiple GPUs use torchrun with --launcher pytorch')
    device = select_device(args.device or settings['device'])
    if distributed:
        if device.type == 'cuda':
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl' if device.type == 'cuda' and os.name != 'nt' else 'gloo')
    if device.type == 'cpu':
        torch.set_num_threads(settings.get('cpu_threads', 4))
    seed = args.seed if args.seed is not None else settings['seed']
    settings['seed'] = seed
    settings['deterministic'] = args.deterministic or settings['deterministic']
    seed_everything(seed, settings['deterministic'])
    batch_size = int(settings['batch_size'])
    if batch_size < world_size or batch_size % world_size:
        raise ValueError('Global train.batch_size must be divisible by GPU/process count')
    if settings['epochs'] < 1 or settings['micro_batch_size'] < 1:
        raise ValueError('epochs and micro_batch_size must be positive')
    if settings['selection_metric'] not in ('psnr', 'ssim'):
        raise ValueError('selection_metric must be psnr or ssim')
    if settings['val_interval'] < 1 or settings['checkpoint_interval'] < 1 or settings['log_interval'] < 1:
        raise ValueError('interval settings must be positive')
    if not (0 < settings['min_lr'] <= settings['lr']) or settings['lr_power'] <= 0:
        raise ValueError('Invalid polynomial learning-rate configuration')
    rows, data_root, audit = prepare_data(cfg['data'], no_validate=args.no_validate)
    subsets = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val')}
    smoke = bool(args.smoke_test)
    if smoke:
        subsets['train'] = subsets['train'][:batch_size * 2]
        subsets['val'] = subsets['val'][:2]
    train_set = PairedDataset(subsets['train'], data_root, cfg['data'], training=True)
    train_set.seed = seed
    val_set = PairedDataset(subsets['val'], data_root, cfg['data'])
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if distributed else None
    loader_generator = torch.Generator()
    local_batch = batch_size // world_size
    train_loader = DataLoader(train_set, batch_size=local_batch, sampler=sampler,
                              shuffle=sampler is None, num_workers=cfg['data']['workers'],
                              pin_memory=device.type == 'cuda', drop_last=False,
                              worker_init_fn=seed_worker, generator=loader_generator,
                              persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=cfg['data']['workers'],
                            pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    work = resolve_path(args.work_dir or cfg['work_dir'])
    if smoke and args.work_dir is None:
        work = work / ('smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    resume_path = resolve_path(args.resume_from) if args.resume_from else None
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
        if (work / 'latest.pth').exists() and resume_path is None:
            raise ValueError('A training run already exists. Use --resume-from {} or choose --work-dir.'.format(work / 'latest.pth'))
    if distributed:
        dist.barrier()
    model = WMNet(**cfg['model']).to(device)
    amp = bool(settings['amp'] and device.type == 'cuda')
    # L1 is weighted by 100; a conservative initial scale avoids overflowing
    # FP16 gradients during the first updates on a randomly initialized model.
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], betas=settings['betas'], weight_decay=settings['weight_decay'])
    start_epoch, global_step, best = 0, 0, float('-inf')
    best_metrics = {}
    total_steps = settings['epochs'] * len(train_loader)
    if resume_path:
        state = load_checkpoint(resume_path)
        if state.get('format') != 'wmnet-training-v1':
            raise ValueError('Not a WM-Net training checkpoint')
        if state['data_fingerprint'] != audit['data_fingerprint']:
            raise ValueError('Paired file listing changed since checkpoint; start a new experiment')
        if state['world_size'] != world_size or state['steps_per_epoch'] != len(train_loader) or state['smoke_test'] != smoke:
            raise ValueError('Resume requires the same world size, loader length and smoke/full mode')
        for key in ('epochs','batch_size','lr','min_lr','lr_power','betas','weight_decay','l1_weight','seed','selection_metric','amp','deterministic'):
            if state['config']['train'][key] != settings[key]:
                raise ValueError('Resume config changed: train.' + key)
        if state['config']['evaluation'] != cfg['evaluation']:
            raise ValueError('Resume evaluation settings changed')
        for key in ('patch_size', 'resize', 'intensity_max'):
            if state['config']['data'][key] != cfg['data'][key]:
                raise ValueError('Resume data preprocessing changed: ' + key)
        if {k:v for k,v in state['config']['model'].items() if k != 'gradient_checkpointing'} != {k:v for k,v in cfg['model'].items() if k != 'gradient_checkpointing'}:
            raise ValueError('Resume architecture changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, global_step, best = state['epoch'], state['global_step'], state['best']
        best_metrics = state.get('best_metrics', {})
        restore_rng(state['rng'])
        if rank == 0 and (work / 'history.csv').exists():
            with (work / 'history.csv').open(encoding='utf-8', newline='') as f:
                history_rows = list(csv.DictReader(f))
            if history_rows and int(history_rows[-1]['epoch']) > start_epoch:
                raise ValueError('Work directory contains later epochs than the resume checkpoint. Use a new --work-dir.')
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                                        broadcast_buffers=False, find_unused_parameters=False)
    raw_model = model.module if distributed else model
    parameters = sum(p.numel() for p in raw_model.parameters())
    log_path = work / 'train.log'

    def log(message):
        if rank == 0:
            line = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + message
            print(line, flush=True)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(line + '\n')

    if rank == 0:
        save_json(cfg, work / 'config.json')
        save_json(audit, work / 'data_summary.json')
        save_json(dict(python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, device=str(device),
                       gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                       parameters=parameters, world_size=world_size, amp=amp,
                       smoke_test=smoke, steps_per_epoch=len(train_loader),
                       train_sampler_padding=(len(sampler)*world_size-len(train_set)) if sampler else 0), work / 'environment.json')
    log('WM-Net: {} parameters; {} train / {} val pairs; {} updates per epoch; global batch {}; micro batch {}; AMP={}'.format(
        parameters, len(train_set), len(val_set), len(train_loader), batch_size, settings['micro_batch_size'], amp))
    if smoke:
        log('SMOKE TEST: provided paired files and configured architecture; at most two training batches and two validation images per epoch.')
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(work / 'tensorboard'), purge_step=start_epoch+1 if resume_path else None)
        except ImportError:
            log('TensorBoard is not installed; CSV and JSON logs remain available.')
    end_epoch = min(settings['epochs'], start_epoch + 1) if smoke else settings['epochs']
    try:
        for epoch in range(start_epoch, end_epoch):
            train_set.epoch = epoch
            loader_generator.manual_seed(seed + epoch)
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            sums = torch.zeros(2, device=device, dtype=torch.float64)
            skipped_updates = 0
            begin = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            initial_weight = raw_model.reconstruction.weight.detach().clone() if smoke else None
            for batch_index, batch in enumerate(train_loader):
                lr = polynomial_lr(global_step, total_steps, settings['lr'], settings['min_lr'], settings['lr_power'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                count = batch['lq'].shape[0]
                micro = min(settings['micro_batch_size'], count)
                for offset in range(0, count, micro):
                    x = batch['lq'][offset:offset+micro].to(device, non_blocking=True)
                    y = batch['gt'][offset:offset+micro].to(device, non_blocking=True)
                    synchronize = offset + micro >= count
                    sync_context = model.no_sync() if distributed and not synchronize else nullcontext()
                    with sync_context:
                        with torch.cuda.amp.autocast(enabled=amp):
                            prediction = model(x)
                        raw_l1 = F.l1_loss(prediction.float(), y.float())
                        if not torch.isfinite(raw_l1):
                            raise RuntimeError('Non-finite L1. Check input intensities; try train.amp=False.')
                        loss = raw_l1 * settings['l1_weight'] * (len(x) / count)
                        scaler.scale(loss).backward()
                    sums[0] += raw_l1.detach().double() * len(x)
                    sums[1] += len(x)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped_updates += int(scaler.get_scale() < old_scale)
                global_step += 1
                if batch_index == 0 or (batch_index + 1) % settings['log_interval'] == 0 or batch_index + 1 == len(train_loader):
                    log('Epoch {}/{} batch {}/{} L1={:.6f} weighted_loss={:.6f} lr={:.8g}'.format(
                        epoch+1, settings['epochs'], batch_index+1, len(train_loader),
                        (sums[0]/sums[1]).item(), (sums[0]/sums[1]).item()*settings['l1_weight'], lr))
            if distributed:
                dist.all_reduce(sums)
            train_l1 = (sums[0] / sums[1]).item()
            if smoke and torch.equal(initial_weight, raw_model.reconstruction.weight.detach()):
                raise RuntimeError('Smoke test made no parameter update. Inspect AMP overflow or gradient flow.')
            if rank == 0:
                val = {}
                if not args.no_validate and ((epoch + 1) % settings['val_interval'] == 0 or epoch + 1 == settings['epochs'] or smoke):
                    val = evaluate(raw_model, val_loader, device, cfg, amp,
                                   preview_path=work / 'validation_preview.png')
                row = dict(epoch=epoch+1, global_step=global_step, lr=lr, train_l1=train_l1,
                           train_loss=train_l1*settings['l1_weight'],
                           val_l1=val.get('l1'), val_loss=val.get('loss'),
                           val_psnr=val.get('psnr'), val_ssim=val.get('ssim'),
                           raw_psnr=val.get('raw_psnr'), raw_ssim=val.get('raw_ssim'),
                           seconds=time.perf_counter()-begin, amp_skipped_updates=skipped_updates,
                           optimizer_updates=len(train_loader)-skipped_updates,
                           peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0)
                history = work / 'history.csv'
                with history.open('a', newline='', encoding='utf-8') as f:
                    out = csv.DictWriter(f, fieldnames=list(row))
                    if f.tell() == 0:
                        out.writeheader()
                    out.writerow(row)
                with (work / 'history.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row) + '\n')
                if writer:
                    for key, value in row.items():
                        if value is not None:
                            writer.add_scalar(key, value, epoch+1)
                    writer.flush()
                improved = bool(val) and val[settings['selection_metric']] > best
                if improved:
                    best = val[settings['selection_metric']]
                best_metrics, improved_metrics = update_best_metrics(best_metrics, val, epoch+1)
                checkpoint = dict(format='wmnet-training-v1', model=raw_model.state_dict(),
                                  optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                  epoch=epoch+1, global_step=global_step, best=best, config=cfg,
                                  data_fingerprint=audit['data_fingerprint'], rng=rng_state(),
                                  world_size=world_size, steps_per_epoch=len(train_loader), smoke_test=smoke,
                                  best_metrics=best_metrics, validation_metrics=val)
                atomic_save(checkpoint, work / 'latest.pth')
                if improved:
                    atomic_save(checkpoint, work / 'best.pth')
                # Compact inference-only snapshots keep both validation optima.
                # Use latest.pth or best.pth for optimizer/RNG-state resume.
                for metric in improved_metrics:
                    snapshot = {k: v for k, v in checkpoint.items() if k not in ('optimizer', 'scaler', 'rng')}
                    snapshot['format'] = 'wmnet-inference-v1'
                    snapshot['selected_by'] = 'validation_' + metric
                    atomic_save(snapshot, work / ('best_' + metric + '.pth'))
                if val:
                    save_json(dict(selection_split='val', best=best_metrics,
                                   note='Each entry contains PSNR and SSIM from the same checkpoint; these are not independent-test results.'),
                              work / 'best_metrics.json')
                if (epoch + 1) % settings['checkpoint_interval'] == 0 or epoch+1 == settings['epochs']:
                    atomic_save(checkpoint, work / ('epoch_{:04d}.pth'.format(epoch+1)))
                log('Epoch {} complete: {}'.format(epoch+1, json.dumps(row)))
            if distributed:
                dist.barrier()
        log('Training finished. Checkpoint: ' + str(work / 'latest.pth'))
    except KeyboardInterrupt:
        log('Interrupted. latest.pth is the last fully completed epoch; resume replays the interrupted epoch.')
        raise
    finally:
        if writer:
            writer.close()
        if distributed:
            dist.destroy_process_group()


def pair_files(noise_dir, clean_dir, split='train'):
    """Minimal I/O adapter: matching relative names; no dataset construction."""
    noise_dir, clean_dir = resolve_path(noise_dir), resolve_path(clean_dir)
    if not noise_dir.is_dir() or not clean_dir.is_dir():
        raise ValueError('Set DATA_CONFIG paths or --{}-noise/--{}-clean to existing paired folders.\nMissing: {} or {}'.format(
            split, split, noise_dir, clean_dir))
    if noise_dir == clean_dir:
        raise ValueError('noise and clean must be different paired directories')
    noisy = {p.relative_to(noise_dir).as_posix(): p for p in noise_dir.rglob('*') if p.suffix.lower() in EXTENSIONS}
    clean = {p.relative_to(clean_dir).as_posix(): p for p in clean_dir.rglob('*') if p.suffix.lower() in EXTENSIONS}
    if noisy.keys() != clean.keys():
        raise ValueError('Unpaired filenames: noise-only {}; clean-only {}'.format(
            sorted(noisy.keys()-clean.keys())[:5], sorted(clean.keys()-noisy.keys())[:5]))
    if not noisy:
        raise ValueError('No supported paired images in ' + str(noise_dir))
    return [dict(split=split, noise=str(noisy[k].resolve()), clean=str(clean[k].resolve())) for k in sorted(noisy)]


def prepare_data(cfg, no_validate=False):
    rows = pair_files(cfg['train_noise'], cfg['train_clean'], 'train')
    if not no_validate:
        validation = pair_files(cfg['val_noise'], cfg['val_clean'], 'val')
        if {r['noise'] for r in rows} & {r['noise'] for r in validation}:
            raise ValueError('Training and validation refer to the same input files')
        rows += validation
    listing = []
    for row in rows:
        fields = [row['split']]
        for key in ('noise','clean'):
            path = Path(row[key])
            stat = path.stat()
            fields.extend((str(path), str(stat.st_size), str(stat.st_mtime_ns)))
        listing.append('|'.join(fields))
    fingerprint = hashlib.sha256('\n'.join(listing).encode()).hexdigest()
    summary = dict(pairs={split:sum(r['split']==split for r in rows) for split in ('train','val')},
                   data_fingerprint=fingerprint,
                   note='User-provided paired folders; no automatic splitting or specimen-level audit.')
    return rows, ROOT, summary


def self_test(cfg, device_name):
    """Full-size synthetic forward/backward check; not an accuracy experiment."""
    device = select_device(device_name or cfg['train']['device'])
    seed_everything(cfg['train']['seed'], True)
    if device.type == 'cpu':
        torch.set_num_threads(cfg['train']['cpu_threads'])
    net = WMNet(**cfg['model']).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['train']['lr'], betas=cfg['train']['betas'])
    amp = bool(device.type == 'cuda' and cfg['train']['amp'])
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.0)
    size = cfg['data']['patch_size']
    target = torch.rand(1,1,size,size,device=device)
    before = net.reconstruction.weight.detach().clone()
    for index in range(cfg['train']['batch_size']):
        noisy = (target + torch.randn_like(target)*0.05).clamp(0,1)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = net(noisy)
        loss = (prediction.float()-target).abs().mean()*cfg['train']['l1_weight']/cfg['train']['batch_size']
        if prediction.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError('Self-test failed: shape or finite loss')
        scaler.scale(loss).backward()
    for block in net.bottleneck:
        grad = block.condition.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum().item() == 0:
            raise RuntimeError('Self-test failed: missing/non-finite conditioning gradients')
    scaler.step(optimizer)
    scaler.update()
    if torch.equal(before,net.reconstruction.weight.detach()):
        raise RuntimeError('Self-test failed: parameters did not update')
    net.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        prediction = net(noisy)
    psnr, ssim = image_metrics(prediction,target,**cfg['evaluation'])
    if not torch.isfinite(psnr).all() or not torch.isfinite(ssim).all():
        raise RuntimeError('Self-test failed: non-finite metrics')
    print('SELF-TEST PASSED: parameters={}, input={}, effective_batch={}, device={}'.format(
        sum(p.numel() for p in net.parameters()),tuple(target.shape),cfg['train']['batch_size'],device))
    print('Verified forward, noise-conditioning gradients, Adam update, PSNR and SSIM. Synthetic input is not a performance result.')


def run_inference(args):
    if not args.checkpoint or not args.input:
        raise ValueError('Inference requires --checkpoint and --input')
    state = load_checkpoint(resolve_path(args.checkpoint))
    cfg = state['config']
    device = select_device(args.device or cfg['train']['device'])
    net = WMNet(**cfg['model']).to(device).eval()
    net.load_state_dict(state['model'], strict=True)
    array = read_grayscale(resolve_path(args.input),cfg['data'].get('intensity_max'))
    x = torch.from_numpy(array)[None,None].to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type=='cuda' and cfg['train']['amp']):
        y = net(x,args.noise_level)
    if not torch.isfinite(y).all():
        raise RuntimeError('Non-finite prediction')
    output = resolve_path(args.output or 'work_dirs/wmnet/prediction.png')
    if output.suffix.lower() not in ('.png','.tif','.tiff'):
        raise ValueError('Use PNG or TIFF for uint16 prediction output')
    if output == resolve_path(args.input):
        raise ValueError('Output must differ from the input image')
    output.parent.mkdir(parents=True,exist_ok=True)
    array = y[0,0].float().clamp(0,1).cpu().numpy()
    Image.fromarray(np.round(array*65535).astype(np.uint16)).save(output)
    np.save(str(output.with_suffix('.npy')),array)
    print('Saved',output,'and',output.with_suffix('.npy'))


def run_evaluation(args):
    if not args.checkpoint or not args.eval_noise or not args.eval_clean:
        raise ValueError('Evaluation requires --checkpoint, --eval-noise and --eval-clean')
    state = load_checkpoint(resolve_path(args.checkpoint))
    cfg = state['config']
    rows = pair_files(args.eval_noise,args.eval_clean,'eval')
    dataset = PairedDataset(rows, ROOT, cfg['data'])
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=cfg['data']['workers'])
    device = select_device(args.device or cfg['train']['device'])
    net = WMNet(**cfg['model']).to(device).eval()
    net.load_state_dict(state['model'],strict=True)
    metrics, records = evaluate(net,loader,device,cfg,
                                amp=device.type=='cuda' and cfg['train']['amp'],per_image=True)
    output = resolve_path(args.output or 'work_dirs/wmnet/evaluation')
    output.mkdir(parents=True,exist_ok=True)
    for key in ('psnr','ssim','raw_psnr','raw_ssim'):
        metrics[key+'_sample_std'] = float(np.std([r[key] for r in records],ddof=1)) if len(records)>1 else None
    metrics.update(num_images=len(records),epoch=state['epoch'],checkpoint=str(args.checkpoint),
                   evaluation_region='fixed center crop',patch_size=cfg['data']['patch_size'],
                   note='Image-level descriptive metrics; not a specimen-level statistical analysis.')
    with (output/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_json(metrics,output/'metrics.json')
    print(json.dumps(metrics,indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description='Standalone two-file WM-Net training')
    parser.add_argument('--mode',choices=['train','evaluate','infer'],default='train')
    for name in ('train-noise','train-clean','val-noise','val-clean','eval-noise','eval-clean'):
        parser.add_argument('--'+name)
    parser.add_argument('--work-dir')
    parser.add_argument('--resume-from')
    parser.add_argument('--checkpoint')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--noise-level',type=float)
    parser.add_argument('--device',help='auto/cpu/cuda/cuda:0')
    parser.add_argument('--epochs',type=int,help='Default 500; overriding changes the experiment')
    parser.add_argument('--batch-size',type=int,help='Global effective batch size; default 4')
    parser.add_argument('--micro-batch-size',type=int,help='Images computed together; default 1')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--no-amp',action='store_true')
    parser.add_argument('--no-validate',action='store_true')
    parser.add_argument('--self-test',action='store_true',help='Full-size synthetic test; no data files needed')
    parser.add_argument('--smoke-test',action='store_true',help='Use up to eight training pairs and two validation pairs for one epoch')
    parser.add_argument('--launcher',choices=['none','pytorch'],default='none')
    parser.add_argument('--local_rank','--local-rank',type=int,default=0)
    parser.add_argument('--gpus',type=int,default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dict(model=copy.deepcopy(MODEL_CONFIG),train=copy.deepcopy(TRAIN_CONFIG),
               data=copy.deepcopy(DATA_CONFIG),evaluation=copy.deepcopy(EVALUATION_CONFIG),work_dir=WORK_DIR)
    for key in ('train_noise','train_clean','val_noise','val_clean','workers'):
        value = getattr(args,key)
        if value is not None:
            cfg['data'][key] = value
    for key in ('epochs','batch_size','micro_batch_size'):
        value = getattr(args,key)
        if value is not None:
            cfg['train'][key] = value
    if args.no_amp:
        cfg['train']['amp'] = False
    if cfg['data']['workers'] < 0 or cfg['data']['patch_size'] < 11:
        raise ValueError('workers must be nonnegative; patch_size must be >=11 for SSIM')
    if args.self_test:
        self_test(cfg,args.device)
    elif args.mode == 'infer':
        run_inference(args)
    elif args.mode == 'evaluate':
        run_evaluation(args)
    else:
        train(cfg,args)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except (ValueError,FileNotFoundError) as error:
        raise SystemExit(str(error))
