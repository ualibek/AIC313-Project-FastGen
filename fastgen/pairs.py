"""Reflow pairs from the flow-matching teacher (Liu et al., 2023; Lee et al., 2024).

Synthetic pairs: noise for each 64-pair group comes from that group's seed (never stored; the trainer
regenerates it), runs through the teacher ODE, and the image is stored as uint8:
    python train.py pairs --snapshot runs/fm_main/snapshots/kimg_030000.pt --ema 2000 --out pairs/<name> \
        --sampler heun --steps 32 --cfg 1.5 --interval 0.2,0.85
Inverted real pairs (--invert): real train images (and their h-flips) run through the same ODE backwards
(t: 0 -> 1); the resulting noise is stored as fp16.

Output: x_XXXX.npy (uint8 [N, 3, 64, 64]), y_XXXX.npy (int16), z_XXXX.npy (fp16, --invert only) and
meta.json. x_XXXX.npy is written last, so it marks a finished shard; reruns skip finished shards.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from fastgen.data import DEFAULT_CACHE, PAIR_SHAPE, group_noise, load_cache, pair_shards
from fastgen.fid import FIDMonitor
from fastgen.flow import sample_euler, sample_heun
from fastgen.unet import NUM_CLASSES, FlowUNet

LABEL_SEED_OFFSET = 1_000_000_000  # per-shard label permutation seed = base_seed + offset + shard
FIXED_META_KEYS = ("mode", "snapshot", "ema", "sampler", "steps", "shift", "cfg", "interval", "group_size",
                   "groups_per_shard", "base_seed", "invert_steps", "flip", "fp32")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate reflow (noise, image) pairs with the teacher ODE")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--ema", default="2000", help="EMA half-life key (kimg) or 'raw'")
    p.add_argument("--out", required=True)
    p.add_argument("--sampler", default="heun", choices=["euler", "heun"])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--interval", default="0.2,0.85", help="guidance interval 'lo,hi', or 'none'")
    p.add_argument("--num_shards", type=int, default=20)
    p.add_argument("--groups_per_shard", type=int, default=800)
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--groups_per_batch", type=int, default=4, help="8 (1024 guided rows) OOMs on 20 GiB")
    p.add_argument("--base_seed", type=int, default=10_000_000)
    p.add_argument("--invert", action="store_true", help="invert real train images instead")
    p.add_argument("--invert_steps", type=int, default=64, help="Heun steps (with final correction) for inversion")
    p.add_argument("--no_flip", action="store_true", help="--invert: skip the h-flipped copies")
    p.add_argument("--fp32", action="store_true", help="run the teacher in fp32 (default bf16 autocast)")
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--no_fid", action="store_true")
    return p.parse_args(argv)


def load_teacher(snapshot, ema, device, compile_net=True):
    snap = torch.load(snapshot, map_location="cpu", weights_only=False)
    net = FlowUNet(sigma_data=snap["sigma_data"], **snap["unet_config"])
    weights = snap["raw"] if ema == "raw" else snap["ema"][ema]
    net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in weights.items()})
    net = net.to(device).eval().requires_grad_(False).to(memory_format=torch.channels_last)
    return torch.compile(net, dynamic=False) if compile_net else net


def guidance_interval(spec):
    return None if spec == "none" else tuple(float(v) for v in spec.split(","))


def teacher_sampler(args):
    """noise -> image with the configured teacher ODE solver (the pair coupling)."""
    fn = sample_euler if args.sampler == "euler" else sample_heun
    kwargs = dict(shift=args.shift, cfg_scale=args.cfg, cfg_interval=guidance_interval(args.interval))
    return lambda net, z, y: fn(net, z, y, args.steps, **kwargs)


def quantize(x):
    """Round to 8 bits; equals evaluate.py's truncation after finalize()'s half-LSB offset."""
    return ((x.float() + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)


def save_npy(path, array):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def write_meta(out, args, extra):
    meta_path = out / "meta.json"
    meta = {"mode": "invert" if args.invert else "synthetic", "snapshot": args.snapshot, "ema": args.ema,
            "sampler": args.sampler, "steps": args.steps, "shift": args.shift, "cfg": args.cfg,
            "interval": args.interval, "group_size": args.group_size, "groups_per_shard": args.groups_per_shard,
            "base_seed": None if args.invert else args.base_seed,
            "invert_steps": args.invert_steps if args.invert else None,
            "flip": (not args.no_flip) if args.invert else None, "fp32": args.fp32, "stored_z": args.invert,
            "label_seed": "base_seed + 1e9 + shard", "shard_fid": {}}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), meta[k]) for k in FIXED_META_KEYS if old.get(k) != meta[k]}
        if diff:
            raise SystemExit(f"{meta_path} was made with different settings: {diff}")
        meta = old
    meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def balanced_subset(labels, per_class=20):
    return np.concatenate([np.flatnonzero(labels == c)[:per_class] for c in range(NUM_CLASSES)])


def generate(args, net, out, device):
    run = teacher_sampler(args)
    gs, groups, gpb = args.group_size, args.groups_per_shard, args.groups_per_batch
    assert groups % gpb == 0, "groups_per_shard must be a multiple of groups_per_batch"
    per_shard = groups * gs
    write_meta(out, args, {"num_shards_target": args.num_shards})
    noise_gen = torch.Generator(device=device)
    done = set(pair_shards(out))
    todo = [s for s in range(args.num_shards) if s not in done]
    print(f"{len(done)} shards done, {len(todo)} to go ({per_shard} pairs each)", flush=True)
    t_start, checked = time.time(), False
    for n_made, s in enumerate(todo, 1):
        label_gen = torch.Generator().manual_seed(args.base_seed + LABEL_SEED_OFFSET + s)
        y = (torch.arange(per_shard) % NUM_CLASSES)[torch.randperm(per_shard, generator=label_gen)]
        xs = torch.empty((per_shard, *PAIR_SHAPE), dtype=torch.uint8)
        for b in range(groups // gpb):
            first = s * groups + b * gpb
            z = torch.cat([group_noise(args.base_seed, k, gs, device, noise_gen) for k in range(first, first + gpb)])
            lo, hi = b * gpb * gs, (b + 1) * gpb * gs
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                x = run(net, z, y[lo:hi].to(device))
            xs[lo:hi] = quantize(x).cpu()
            if (b + 1) % 50 == 0:
                print(f"[pairs] shard {s}: {hi}/{per_shard}", flush=True)
        save_npy(out / f"y_{s:04d}.npy", y.numpy().astype(np.int16))
        save_npy(out / f"x_{s:04d}.npy", xs.numpy())
        elapsed = time.time() - t_start
        rate = n_made * per_shard / elapsed
        eta_h = (len(todo) - n_made) * per_shard / rate / 3600
        print(f"[pairs] shard {s} done: {rate:.1f} pairs/s, elapsed {elapsed / 3600:.2f} h, ETA {eta_h:.2f} h, "
              f"peak mem {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
        if not checked and not args.no_fid:
            checked = True
            idx = balanced_subset(y.numpy())
            monitor = FIDMonitor(device, num_workers=4)
            monitor.load_or_build_reference(load_cache(args.cache))
            fid = monitor.fid(xs[idx])
            del monitor
            meta = json.loads((out / "meta.json").read_text())
            meta["shard_fid"][str(s)] = fid
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[pairs] shard {s}: train-reference FID of a balanced 20/class subset = {fid:.3f}", flush=True)


def invert(args, net, out, device):
    cache = load_cache(args.cache)
    images, labels = cache["images"], cache["labels"]
    if not args.no_flip:
        images, labels = torch.cat([images, images.flip(3)]), torch.cat([labels, labels])
    n = len(images)
    batch = args.groups_per_batch * args.group_size
    grid = torch.linspace(0, 1, args.invert_steps + 1, dtype=torch.float64).tolist()
    kwargs = dict(cfg_scale=args.cfg, cfg_interval=guidance_interval(args.interval))
    zs = torch.empty((n, *PAIR_SHAPE), dtype=torch.float16)
    t_start = time.time()
    for i in range(0, n, batch):
        x = images[i:i + batch].to(device).float() / 127.5 - 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
            z = sample_heun(net, x, labels[i:i + batch].to(device), args.invert_steps, t_grid=grid,
                            final_correction=True, **kwargs)
        zs[i:i + batch] = z.float().half().cpu()
        if (i // batch) % 10 == 0:
            print(f"[invert] {i + len(x)}/{n} ({(i + len(x)) / (time.time() - t_start):.1f} img/s)", flush=True)

    # Diagnostics: inverted noise vs N(0, I) (norm / sqrt(d) ~ 1, neighbour correlation ~ 0) and
    # reconstruction through the forward (pair-generating) sampler.
    zf = zs.float()
    norms = zf.flatten(1).norm(dim=1) / zf[0].numel() ** 0.5
    a, b = zf[..., :-1].flatten(1), zf[..., 1:].flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    lag1 = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean().item()
    run = teacher_sampler(args)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:256]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
        rec = torch.cat([run(net, zf[i].to(device), labels[i].to(device)) for i in idx.split(batch)])
    real = images[idx].to(device).float() / 127.5 - 1
    mse = (quantize(rec).float() / 127.5 - 1 - real).pow(2).flatten(1).mean(1)
    psnr = (10 * torch.log10(4 / mse.clamp(min=1e-10))).mean().item()
    stats = {"num_pairs": n, "z_mean": zf.mean().item(), "z_std": zf.std().item(),
             "norm_over_sqrt_d_mean": norms.mean().item(), "norm_over_sqrt_d_std": norms.std().item(),
             "lag1_autocorr": lag1, "reconstruction_psnr_db": psnr, "seconds": time.time() - t_start}
    print(f"[invert] {stats}", flush=True)
    write_meta(out, args, {"invert_stats": stats})
    save_npy(out / "y_0000.npy", labels.numpy().astype(np.int16))
    save_npy(out / "z_0000.npy", zs.numpy())
    save_npy(out / "x_0000.npy", images.numpy())


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_teacher(args.snapshot, args.ema, device, compile_net=not args.no_compile)
    if args.invert:
        if pair_shards(out):
            raise SystemExit(f"{out} already holds an inverted pair set")
        invert(args, net, out, device)
    else:
        generate(args, net, out, device)


if __name__ == "__main__":
    main()
