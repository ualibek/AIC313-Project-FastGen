"""Distribution Matching Distillation, DMD2 (Yin et al., 2024), in the flow parametrization.

Path x_t = (1 - t) x + t eps (EDM's sigma = t / (1 - t)); x-prediction x_hat = x_t - t v.
Details (timestep ranges, gradient normalization, loss reductions, classifier head) follow the official
implementation (github.com/tianweiy/DMD2, main/edm/edm_guidance.py); no code was copied.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastgen.unet import NULL_CLASS

SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0  # EDM's Karras schedule, as used by DMD2
DM_RANGE = (0.02, 0.98)  # DMD2's min/max_step_percent for the distribution-matching loss


def karras_t(n, device, generator, lo=0.0, hi=1.0):
    """t = sigma / (1 + sigma), sigma on the Karras schedule at a uniform fraction u in [lo, hi] (u = 0 is sigma_min).

    DMD2 draws a uniform integer index of a 1000-step Karras grid; this is the continuous version.
    """
    u = lo + (hi - lo) * torch.rand(n, device=device, generator=generator, dtype=torch.float64)
    a, b = SIGMA_MIN ** (1 / RHO), SIGMA_MAX ** (1 / RHO)
    sigma = (a + u * (b - a)) ** RHO
    return (sigma / (1 + sigma)).float()


def add_noise(x, t, noise):
    tb = t[:, None, None, None]
    return (1 - tb) * x + tb * noise


def guided_velocity(net, x, t, class_labels, cfg_scale=1.0, cfg_interval=None):
    """Velocity at per-sample times t with classifier-free guidance applied only where t is in cfg_interval.

    With guidance on, every call runs the batched cond + uncond pair (one fixed shape for torch.compile); samples
    outside the interval take the conditional velocity.
    """
    if cfg_scale == 1.0:
        return net(x, t, class_labels=class_labels).float()
    null = torch.full_like(class_labels, NULL_CLASS)
    out = net(torch.cat([x, x]), torch.cat([t, t]), class_labels=torch.cat([class_labels, null])).float()
    v_cond, v_uncond = out.chunk(2)
    guided = v_uncond + cfg_scale * (v_cond - v_uncond)
    if cfg_interval is None:
        return guided
    lo, hi = cfg_interval
    inside = ((t >= lo) & (t <= hi))[:, None, None, None]
    return torch.where(inside, guided, v_cond)


class DiscHead(nn.Module):
    """DMD2's GAN classifier on the fake score's 8x8 bottleneck: 4x4 convs 8 -> 4 -> 1 with GroupNorm + SiLU,
    then a 1x1 conv to one logit per sample."""

    def __init__(self, channels=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 4, stride=2, padding=1), nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 4, stride=4, padding=0), nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, h):
        return self.net(h).flatten()


def dm_loss(x_g, x_real, x_fake):
    """Distribution matching loss on generator samples x_g given the real / fake scores' x-predictions.

    The gradient w.r.t. x_g is (x_fake - x_real) / mean|x_g - x_real| (per sample) / numel, written as
    0.5 * MSE(x_g, stopgrad(x_g - g)) as in DMD2. Returns the loss and the per-sample RMS of g (for logging).
    """
    with torch.no_grad():
        p_real = x_g - x_real
        grad = (x_fake - x_real) / p_real.abs().mean(dim=(1, 2, 3), keepdim=True)
        grad = torch.nan_to_num(grad)
    loss = 0.5 * F.mse_loss(x_g, (x_g - grad).detach())
    return loss, grad.pow(2).mean(dim=(1, 2, 3)).sqrt()


def gan_d_loss(logit_real, logit_fake):
    """Non-saturating discriminator loss (DMD2's guidance-turn classification loss)."""
    return F.softplus(logit_fake).mean() + F.softplus(-logit_real).mean()


def gan_g_loss(logit_fake):
    return F.softplus(-logit_fake).mean()
