"""Pixel-space UNet backbone for flow matching.

Our own implementation in the style of the DDPM++ / ADM UNets as configured in
EDM (Karras et al., 2022; Song et al., 2021; Dhariwal & Nichol, 2021):
residual blocks with adaptive GroupNorm conditioning, avg-pool / nearest
resampling blocks, skip concatenation, and multi-head self-attention at low
resolutions.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_CLASSES = 151
NULL_CLASS = NUM_CLASSES  # extra embedding row used for classifier-free guidance
AUG_DIM = 3  # (translate_x, translate_y, log2 scale) of the non-leaking augment pipe

# Frozen architecture configs. Checkpoints are only loadable with the config they were trained with.
UNET_CONFIGS = {
    # Chosen config: ADM/EDM2-depth (3 blocks/level), 8x8 level widened to 384 ch. ~79.4M params.
    "unet79": dict(model_channels=128, channel_mult=(1, 2, 2, 3), num_blocks=3, attn_res=(16, 8), dropout=0.15),
    # EDM's 64x64 DDPM++ config (FFHQ/AFHQv2-64) + class/augment conditioning. ~68.1M params.
    "unet68": dict(model_channels=128, channel_mult=(1, 2, 2, 2), num_blocks=4, attn_res=(16,), dropout=0.15),
}


def _zero_(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class PositionalEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar. Frequencies span [1/max_positions, 1]."""

    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        half = self.num_channels // 2
        freqs = torch.arange(half, dtype=torch.float32, device=x.device) / half
        freqs = (1.0 / self.max_positions) ** freqs
        x = x.float()[:, None] * freqs[None, :]
        return torch.cat([x.cos(), x.sin()], dim=1)


class Attention(nn.Module):
    """Multi-head self-attention over spatial positions (64 channels per head)."""

    def __init__(self, channels, head_channels=64, eps=1e-6):
        super().__init__()
        self.num_heads = max(channels // head_channels, 1)
        self.norm = nn.GroupNorm(32, channels, eps=eps)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = _zero_(nn.Conv2d(channels, channels, 1))
        # "sdpa" uses fused kernels; "math" is a plain implementation that supports forward-mode AD
        # (torch.func.jvp), which the fused kernels do not in torch 2.6.
        self.attn_impl = "sdpa"

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(1)  # each [b, heads, hw, d]
        if self.attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            scale = 1.0 / math.sqrt(q.shape[-1])
            weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
            out = torch.matmul(weights, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, *, up=False, down=False, attention=False, dropout=0.0, eps=1e-6):
        super().__init__()
        self.up, self.down, self.dropout = up, down, dropout
        self.norm0 = nn.GroupNorm(32, in_ch, eps=eps)
        self.conv0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.affine = nn.Linear(emb_ch, out_ch * 2)
        self.norm1 = nn.GroupNorm(32, out_ch, eps=eps)
        self.conv1 = _zero_(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.attn = Attention(out_ch, eps=eps) if attention else None

    def _resample(self, x):
        if self.up:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            return F.avg_pool2d(x, 2)
        return x

    def forward(self, x, emb):
        h = self.conv0(self._resample(F.silu(self.norm0(x))))
        scale, shift = self.affine(emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(torch.addcmul(shift, self.norm1(h), scale + 1))
        h = self.conv1(F.dropout(h, p=self.dropout, training=self.training))
        x = self._resample(x)
        if self.skip is not None:
            x = self.skip(x)
        x = (h + x) * math.sqrt(0.5)
        if self.attn is not None:
            x = (x + self.attn(x)) * math.sqrt(0.5)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        img_resolution=64,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 3),
        num_blocks=3,
        attn_res=(16, 8),
        dropout=0.15,
        num_classes=NUM_CLASSES,
        aug_dim=AUG_DIM,
        time_scale=1.0,
    ):
        super().__init__()
        emb_ch = model_channels * 4
        self.time_scale = time_scale
        # Conditioning (EDM-style): noise embedding + class embedding + augment-label embedding,
        # summed in the positional-embedding space, then a 2-layer MLP.
        self.map_noise = PositionalEmbedding(model_channels)
        self.map_label = nn.Embedding(num_classes + 1, model_channels)  # last row = null class
        nn.init.normal_(self.map_label.weight, std=1.0)
        self.map_augment = nn.Linear(aug_dim, model_channels, bias=False) if aug_dim else None
        if self.map_augment is not None:
            nn.init.zeros_(self.map_augment.weight)  # no effect until learned
        self.map_layer0 = nn.Linear(model_channels, emb_ch)
        self.map_layer1 = nn.Linear(emb_ch, emb_ch)

        ch = model_channels * channel_mult[0]
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        skips = [ch]
        res = img_resolution
        self.enc = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            if level > 0:
                self.enc.append(ResBlock(ch, ch, emb_ch, down=True, dropout=dropout))
                res //= 2
                skips.append(ch)
            for _ in range(num_blocks):
                out = model_channels * mult
                self.enc.append(ResBlock(ch, out, emb_ch, attention=res in attn_res, dropout=dropout))
                ch = out
                skips.append(ch)
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_ch, attention=True, dropout=dropout),
            ResBlock(ch, ch, emb_ch, dropout=dropout),
        ])
        self.dec = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_blocks + 1):
                out = model_channels * mult
                self.dec.append(ResBlock(ch + skips.pop(), out, emb_ch, attention=res in attn_res, dropout=dropout))
                ch = out
            if level > 0:
                self.dec.append(ResBlock(ch, ch, emb_ch, up=True, dropout=dropout))
                res *= 2
        self.norm_out = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_out = _zero_(nn.Conv2d(ch, out_channels, 3, padding=1))

    def set_attn_impl(self, impl):
        for module in self.modules():
            if isinstance(module, Attention):
                module.attn_impl = impl

    def forward(self, x, t, class_labels, aug_labels=None):
        emb = self.map_noise(t * self.time_scale) + self.map_label(class_labels)
        if self.map_augment is not None and aug_labels is not None:
            emb = emb + self.map_augment(aug_labels.to(emb.dtype))
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        h = self.conv_in(x)
        hs = [h]
        for block in self.enc:
            h = block(h, emb)
            hs.append(h)
        for block in self.mid:
            h = block(h, emb)
        for block in self.dec:
            if not block.up:
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)
        return self.conv_out(F.silu(self.norm_out(h)))


class FlowUNet(nn.Module):
    """Velocity predictor v(x_t, t, y) for the path x_t = (1 - t) x0 + t eps (t = 1 is noise).

    The input is scaled to unit variance with c_in(t) = 1 / sqrt((1 - t)^2 sigma_d^2 + t^2)
    (EDM-style preconditioning). class_labels=None selects the null (unconditional) class.
    """

    def __init__(self, sigma_data=0.65, **unet_kwargs):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.unet = UNet(**unet_kwargs)

    def forward(self, x, t, class_labels=None, aug_labels=None):
        b = x.shape[0]
        t = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1).expand(b)
        if class_labels is None:
            class_labels = torch.full((b,), NULL_CLASS, device=x.device, dtype=torch.long)
        c_in = torch.rsqrt((1 - t) ** 2 * self.sigma_data ** 2 + t ** 2)
        return self.unet(x * c_in[:, None, None, None].to(x.dtype), t, class_labels, aug_labels)


def build_flow_unet(arch, sigma_data):
    return FlowUNet(sigma_data=sigma_data, **UNET_CONFIGS[arch])
