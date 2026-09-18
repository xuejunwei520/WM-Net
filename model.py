import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=128, scale=1000.0):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError('embedding_dim must be even and >= 4')
        frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2).float() / (dim // 2 - 1))
        self.register_buffer('frequency', frequency)
        self.scale = scale

    def forward(self, noise):
        phase = noise.float().reshape(-1, 1) * self.scale * self.frequency[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


def norm(channels, groups):
    return nn.GroupNorm(math.gcd(groups, channels), channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            norm(out_channels, groups), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            norm(out_channels, groups), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels, embedding_dim, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = norm(channels, groups)
        self.condition = nn.Linear(embedding_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = norm(channels, groups)
        self.act = nn.SiLU()

    def forward(self, x, embedding):
        h = self.norm1(self.conv1(x))
        h = self.act(h + self.condition(embedding)[:, :, None, None])
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class DecoderLevel(nn.Module):
    def __init__(self, in_channels, skip_channels, groups):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, skip_channels, 1)
        self.fusion = DoubleConv(2 * skip_channels, skip_channels, groups)

    def forward(self, x, skip):
        x = self.projection(F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False))
        return self.fusion(torch.cat((x, skip), dim=1))


class WMNet(nn.Module):
    """Four encoder/decoder levels, six conditioned residual blocks, no sampling.

    Default noise estimate: per-image RMS residual from a 3x3 local mean,
    computed only from the input (never the reference). This is an explicit
    engineering choice, not a speckle estimator specified by the manuscript.
    The caller may supply measured descriptors of shape [B] in [0,1].
    """
    def __init__(self, in_channels=1, out_channels=1,
                 channels=(64, 128, 256, 512), num_res_blocks=6,
                 embedding_dim=128, embedding_hidden=512, norm_groups=8,
                 noise_scale=1000.0, gradient_checkpointing=False):
        super().__init__()
        if len(channels) != 4 or in_channels != 1 or out_channels != 1:
            raise ValueError('WM-Net requires four levels and grayscale input/output')
        self.gradient_checkpointing = gradient_checkpointing
        self._checkpoint_nonreentrant = 'use_reentrant' in inspect.signature(checkpoint).parameters
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels if i == 0 else channels[i-1], c, norm_groups)
            for i, c in enumerate(channels)])
        self.pool = nn.AvgPool2d(2)
        self.noise_embedding = nn.Sequential(
            SinusoidalEmbedding(embedding_dim, noise_scale),
            nn.Linear(embedding_dim, embedding_hidden), nn.SiLU(),
            nn.Linear(embedding_hidden, embedding_hidden), nn.SiLU())
        self.bottleneck = nn.ModuleList([
            ConditionedResidualBlock(channels[-1], embedding_hidden, norm_groups)
            for _ in range(num_res_blocks)])
        # The deepest decoder level fuses the 512-channel encoder skip without
        # upsampling. The next three levels restore H/4, H/2 and H resolution.
        self.decoder_deep = DoubleConv(2 * channels[-1], channels[-1], norm_groups)
        self.decoder = nn.ModuleList([
            DecoderLevel(channels[i+1], channels[i], norm_groups) for i in (2, 1, 0)])
        self.reconstruction = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    @staticmethod
    @torch.no_grad()
    def estimate_noise(x):
        x = x.detach().float()
        smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
        return (x - smooth).square().mean((1, 2, 3)).sqrt().clamp(0, 1)

    def _run(self, module, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if self._checkpoint_nonreentrant:
                return checkpoint(module, *args, use_reentrant=False)
            # PyTorch 1.10 reentrant checkpoint needs one grad-requiring input.
            return checkpoint(module, *args)
        return module(*args)

    def forward(self, x, noise_level=None):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected [B,1,H,W], H and W >= 8')
        if noise_level is None:
            noise_level = self.estimate_noise(x)
        else:
            noise_level = torch.as_tensor(noise_level, device=x.device, dtype=torch.float32).reshape(-1)
            if noise_level.numel() == 1:
                noise_level = noise_level.expand(x.shape[0])
            if noise_level.numel() != x.shape[0] or not torch.isfinite(noise_level).all() or ((noise_level < 0) | (noise_level > 1)).any():
                raise ValueError('noise_level must contain B finite values in [0,1]')
        embedding = self.noise_embedding(noise_level)
        if self.gradient_checkpointing and self.training and not self._checkpoint_nonreentrant:
            x = x.detach().requires_grad_(True)
        skips = []
        for i, layer in enumerate(self.encoder):
            x = self._run(layer, self.pool(x) if i else x)
            skips.append(x)
        for block in self.bottleneck:
            x = self._run(block, x, embedding)
        x = self._run(self.decoder_deep, torch.cat((x, skips[-1]), dim=1))
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            x = self._run(layer, x, skip)
        # A linear reconstruction head avoids clipping gradients during training.
        return self.reconstruction(x)

    def init_weights(self, pretrained=None):
        """Compatibility with MMEditing BasicRestorer; PyTorch initializes layers."""
        if pretrained is not None:
            options = {'map_location': 'cpu'}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                options['weights_only'] = False
            # Load only trusted checkpoints; training states use pickle objects.
            state = torch.load(str(pretrained), **options)
            state = state.get('model', state.get('state_dict', state))
            self.load_state_dict({k[10:] if k.startswith('generator.') else k: v for k, v in state.items()})
