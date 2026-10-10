"""Rectified-flow / flow-matching objective and samplers.

Path: x_t = (1 - t) x0 + t eps, t in [0, 1], t = 1 is pure noise; velocity v = eps - x0.
Sampling integrates dx/dt = v from t = 1 to t = 0.
"""

import torch

from fastgen.unet import NULL_CLASS


def sample_t(n, device, generator, mean=-0.8, std=1.6, p_uniform=0.1):
    """Mixture of LogitNormal(mean, std) (= log-normal sigma = t / (1 - t)) and U(0, 1)."""
    z = torch.randn(n, device=device, generator=generator)
    t = torch.sigmoid(mean + std * z)
    u = torch.rand(n, device=device, generator=generator)
    use_u = torch.rand(n, device=device, generator=generator) < p_uniform
    return torch.where(use_u, u, t)


def loss_weight(t, sigma_data):
    """EDM's loss weighting expressed for v-prediction: w(t) = (t^2 + sigma_d^2 (1 - t)^2) / sigma_d^2."""
    return (t ** 2 + sigma_data ** 2 * (1 - t) ** 2) / sigma_data ** 2


def fm_loss(net, x0, t, noise, class_labels, aug_labels, sigma_data):
    """Per-sample weighted loss and per-sample unweighted v-MSE (for logging)."""
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * noise
    target = noise - x0
    pred = net(xt, t, class_labels=class_labels, aug_labels=aug_labels)
    mse = (pred.float() - target).pow(2).mean(dim=(1, 2, 3))
    return mse * loss_weight(t, sigma_data), mse


def time_grid(steps, shift=1.0, device="cpu"):
    """Decreasing times 1 -> 0; shift > 1 spends more steps at high noise (SD3-style shift)."""
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float64)
    t = shift * t / (1 + (shift - 1) * t)
    return t.float()


class NFECounter:
    """Counts network evaluations as (rows processed) / (batch size): a batched CFG pair counts 2."""

    def __init__(self, net, batch_size):
        self.net, self.batch_size, self.nfe = net, batch_size, 0.0

    def __call__(self, x, t, **kwargs):
        self.nfe += x.shape[0] / self.batch_size
        return self.net(x, t, **kwargs)


def velocity(net, x, t, class_labels, cfg_scale=1.0, cfg_interval=None):
    """Velocity at scalar time t, with optional classifier-free guidance (batched cond + uncond)."""
    b = x.shape[0]
    tt = torch.full((b,), float(t), device=x.device)
    guided = cfg_scale != 1.0 and class_labels is not None
    if guided and cfg_interval is not None:
        lo, hi = cfg_interval
        guided = lo <= float(t) <= hi
    if not guided:
        return net(x, tt, class_labels=class_labels)
    null = torch.full_like(class_labels, NULL_CLASS)
    out = net(torch.cat([x, x]), torch.cat([tt, tt]), class_labels=torch.cat([class_labels, null]))
    v_cond, v_uncond = out.chunk(2)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_euler(net, noise, class_labels, steps, shift=1.0, cfg_scale=1.0, cfg_interval=None):
    ts = time_grid(steps, shift, noise.device)
    x = noise
    for i in range(steps):
        v = velocity(net, x, ts[i], class_labels, cfg_scale, cfg_interval)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


@torch.no_grad()
def sample_heun(net, noise, class_labels, steps, shift=1.0, cfg_scale=1.0, cfg_interval=None):
    """Heun (2nd order) with a final Euler step to t = 0; uses 2 * steps - 1 evaluations."""
    ts = time_grid(steps, shift, noise.device)
    x = noise
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        v = velocity(net, x, t, class_labels, cfg_scale, cfg_interval)
        x_euler = x + (t_next - t) * v
        if i == steps - 1:
            x = x_euler
        else:
            v_next = velocity(net, x_euler, t_next, class_labels, cfg_scale, cfg_interval)
            x = x + (t_next - t) * 0.5 * (v + v_next)
    return x


def finalize(x, half_lsb=True):
    """Output post-processing shared by the FID monitor and Model.sample().

    evaluate.py quantizes with a truncating byte cast; adding half an 8-bit step (1/255 in
    [-1, 1] units) turns that into round-to-nearest (e.g. white stays 255 instead of 254).
    """
    return x + 1.0 / 255 if half_lsb else x
