"""Training loop for the pixel-space velocity UNet.

Stage 1 (default): class-conditional flow-matching (1-rectified flow) teacher on the train split.
Reflow (--pairs DIR): 2-rectified flow on teacher-generated (noise, image) pairs, initialized from the
teacher (--init_snapshot), with the U-shaped timesteps and pseudo-Huber loss of Lee et al. (2024).
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from fastgen.data import (DEFAULT_CACHE, GPUData, PairData, augment, balanced_classes, grid_classes, load_cache,
                          to_eval_uint8)
from fastgen.ema import EMA
from fastgen.fid import FIDMonitor, memorization_ratio
from fastgen.flow import finalize, fm_loss, reflow_loss, sample_euler, sample_t, sample_t_ushape
from fastgen.unet import NULL_CLASS, UNET_CONFIGS, build_flow_unet
from src.utils import count_parameters

VRAM_LIMIT_BYTES = 20e9  # README: 20 GB peak training VRAM
GRID_STEPS = (0, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000)
FID_BATCH = 302  # 3020 = 10 x 302, one compiled shape


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the flow-matching teacher")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--arch", default="unet79", choices=sorted(UNET_CONFIGS))
    p.add_argument("--sigma_data", type=float, default=0.65)
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--micro_batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--total_kimg", type=float, default=30000)
    p.add_argument("--time_budget_h", type=float, default=None,
                   help="stop after this many hours of pure training time (excludes evals)")
    p.add_argument("--ema_halflives_kimg", default="500,2000,5000")
    p.add_argument("--p_uncond", type=float, default=0.1)
    p.add_argument("--p_aug", type=float, default=0.15)
    p.add_argument("--t_mean", type=float, default=-0.8)
    p.add_argument("--t_std", type=float, default=1.6)
    p.add_argument("--p_uniform", type=float, default=0.1)
    p.add_argument("--t_dist", default="logitnormal", choices=["logitnormal", "ushape"])
    p.add_argument("--t_a", type=float, default=4.0, help="U-shaped t: p(t) ~ cosh(a (t - 1/2))")
    p.add_argument("--loss", default="edm_mse", choices=["edm_mse", "huber", "mse"],
                   help="edm_mse for flow matching; huber / mse (velocity residual) for reflow")
    p.add_argument("--pairs", default=None, help="reflow pair set directory (fastgen/pairs.py)")
    p.add_argument("--init_snapshot", default=None, help="initialize weights from this snapshot (not on resume)")
    p.add_argument("--init_ema", default="2000", help="EMA key of --init_snapshot, or 'raw'")
    p.add_argument("--snapshot_every_kimg", type=float, default=3000)
    p.add_argument("--fid_every_kimg", type=float, default=1500)
    p.add_argument("--fid_sampler", default="32", help="Euler sampler spec scored for every EMA (see parse_sampler)")
    p.add_argument("--fid_extra", default="1,4", help="comma-separated sampler specs scored for the best EMA")
    p.add_argument("--grid_samplers", default="1,4,32", help="comma-separated sampler specs for sample grids")
    p.add_argument("--fid_per_class", type=int, default=20)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--latest_every_min", type=float, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if (args.pairs is None) != (args.loss == "edm_mse"):
        p.error("--loss edm_mse is for flow matching; reflow (--pairs) uses huber or mse")
    return args


def parse_sampler(spec):
    """Euler sampler spec: 'N' = N uniform steps; 'N@a-b' = explicit grid (1, a, b, 0), e.g. '2@0.8'."""
    steps, _, mids = spec.partition("@")
    steps = int(steps)
    kwargs = {}
    if mids:
        inner = [float(v) for v in mids.split("-")]
        assert len(inner) == steps - 1, f"sampler spec {spec!r} needs {steps - 1} intermediate times"
        kwargs["t_grid"] = (1.0, *inner, 0.0)
    suffix = f"@{mids}" if mids else ""
    return {"name": f"euler{steps}{suffix}", "file": f"euler{steps:02d}{suffix.replace('@', '_t')}",
            "steps": steps, "kwargs": kwargs}


class CSVLog:
    def __init__(self, path, fields):
        self.path, self.fields = Path(path), fields
        new = not self.path.exists()
        self.file = self.path.open("a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if new:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({k: row.get(k, "") for k in self.fields})
        self.file.flush()


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    latest_path = run_dir / "latest.pt"
    if latest_path.exists() and not args.resume:
        raise SystemExit(f"{latest_path} exists; pass --resume or use a new --run_dir")

    cache = load_cache(args.cache)
    data = GPUData(cache, device)  # real train images: flow-matching data, memorization metric, real grid
    pairs = PairData(args.pairs, device, seed=args.seed) if args.pairs else None
    epoch_size = pairs.num if pairs else data.num
    if pairs:
        print(f"Reflow pairs: {pairs.num:,} from {args.pairs} ({'stored' if pairs.stored_z else 'seeded'} noise)")
    halflives = [float(h) for h in args.ema_halflives_kimg.split(",")]
    assert args.batch % args.micro_batch == 0
    accum = args.batch // args.micro_batch

    net = build_flow_unet(args.arch, args.sigma_data).to(device).to(memory_format=torch.channels_last)
    num_params = count_parameters(net, trainable_only=False)
    print(f"{args.arch}: {num_params:,} parameters")
    assert num_params <= 100_000_000
    if args.init_snapshot and not (args.resume and latest_path.exists()):
        snap = torch.load(args.init_snapshot, map_location="cpu", weights_only=False)
        assert snap["arch"] == args.arch and snap["sigma_data"] == args.sigma_data, "init snapshot config mismatch"
        weights = snap["raw"] if args.init_ema == "raw" else snap["ema"][args.init_ema]
        net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in weights.items()})
        del snap, weights
        print(f"Initialized from {args.init_snapshot} [ema={args.init_ema}]")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    emas = [EMA(net, hl) for hl in halflives]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    state = {"step": 0, "nimg": 0, "train_sec": 0.0, "wall_sec": 0.0}
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        for ema, sd in zip(emas, ckpt["emas"]):
            ema.load_state_dict(sd)
        gen.set_state(ckpt["gen"].cpu())
        state = ckpt["state"]
        del ckpt
        if pairs:
            pairs.perm_gen.manual_seed(args.seed + state["step"])  # fresh pair order instead of replaying epoch 1
        print(f"Resumed from {latest_path} at step {state['step']}, {state['nimg'] / 1e6:.2f} Mimg")
    else:
        config = vars(args) | {"num_params": num_params, "unet_config": UNET_CONFIGS[args.arch],
                               "data_stats": cache["stats"], "pairs_meta": pairs.meta if pairs else None}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    cnet = net if args.no_compile else torch.compile(net)
    # Separate inference copy (EMA weights get copied in) so training graphs are never re-specialized.
    eval_net = build_flow_unet(args.arch, args.sigma_data).to(device).to(memory_format=torch.channels_last).eval()
    eval_net.requires_grad_(False)
    ceval = eval_net if args.no_compile else torch.compile(eval_net, dynamic=False)

    monitor = FIDMonitor(device, num_workers=4)
    monitor.load_or_build_reference(cache)

    grid_y = grid_classes().to(device)
    grid_noise = torch.randn(len(grid_y), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(123))
    if not (run_dir / "samples" / "real_grid.png").exists():
        firsts = torch.stack([torch.nonzero(data.labels == c)[0, 0] for c in grid_y])
        save_image(data.images[firsts].float() / 255, run_dir / "samples" / "real_grid.png", nrow=8)
    if pairs and not (run_dir / "samples" / "pairs_grid.png").exists():
        save_image(pairs.x[:64].float() / 255, run_dir / "samples" / "pairs_grid.png", nrow=8)
    fid_main = parse_sampler(args.fid_sampler)
    fid_extra = [parse_sampler(spec) for spec in args.fid_extra.split(",") if spec]
    grid_samplers = [parse_sampler(spec) for spec in args.grid_samplers.split(",") if spec]

    fid_y = balanced_classes(per_class=args.fid_per_class).to(device)
    fid_noise = torch.randn(len(fid_y), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(1234))

    train_log = CSVLog(run_dir / "log.csv", [
        "step", "kimg", "epoch", "loss", "mse_t0", "mse_t1", "mse_t2", "mse_t3", "grad_norm", "lr",
        "img_per_sec", "peak_mem_gib", "train_h", "wall_h"])
    fid_log = CSVLog(run_dir / "fid.csv", ["step", "kimg", "train_h", "ema_kimg", "sampler", "nfe", "fid", "memorized"])

    def generate(model, noise, labels, sampler, batch):
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(labels), batch):
                x = sample_euler(model, noise[i:i + batch], labels[i:i + batch], sampler["steps"], **sampler["kwargs"])
                out.append(finalize(x.float()))
        return torch.cat(out)

    def save_grids():
        emas[0].copy_to(eval_net)
        kimg = state["nimg"] // 1000
        for sampler in grid_samplers:
            x = generate(ceval, grid_noise, grid_y, sampler, len(grid_y))
            save_image(to_eval_uint8(x).float() / 255,
                       run_dir / "samples" / f"kimg{kimg:06d}_{sampler['file']}.png", nrow=8)

    def run_fid():
        t0 = time.time()
        kimg = state["nimg"] / 1000
        best = None
        for hl, ema in zip(halflives, emas):
            ema.copy_to(eval_net)
            x = generate(ceval, fid_noise, fid_y, fid_main, FID_BATCH)
            fid = monitor.fid(to_eval_uint8(x))
            mem = memorization_ratio(x, data.images)
            fid_log.write({"step": state["step"], "kimg": f"{kimg:.0f}", "train_h": f"{state['train_sec'] / 3600:.3f}",
                           "ema_kimg": hl, "sampler": fid_main["name"], "nfe": fid_main["steps"],
                           "fid": f"{fid:.3f}", "memorized": f"{mem:.4f}"})
            print(f"[fid] kimg {kimg:.0f} ema {hl:g}k {fid_main['name']}: FID {fid:.2f}, memorized {mem:.3%}")
            if best is None or fid < best[0]:
                best = (fid, hl, ema)
        best[2].copy_to(eval_net)
        for sampler in fid_extra:
            x = generate(ceval, fid_noise, fid_y, sampler, FID_BATCH)
            fid = monitor.fid(to_eval_uint8(x))
            fid_log.write({"step": state["step"], "kimg": f"{kimg:.0f}", "train_h": f"{state['train_sec'] / 3600:.3f}",
                           "ema_kimg": best[1], "sampler": sampler["name"], "nfe": sampler["steps"],
                           "fid": f"{fid:.3f}"})
            print(f"[fid] kimg {kimg:.0f} ema {best[1]:g}k {sampler['name']}: FID {fid:.2f}")
        print(f"[fid] eval took {time.time() - t0:.0f}s")
        return best[0]

    def save_snapshot():
        kimg = round(state["nimg"] / 1000)
        snap = {"arch": args.arch, "sigma_data": args.sigma_data, "unet_config": UNET_CONFIGS[args.arch],
                "nimg": state["nimg"], "step": state["step"], "raw": net.state_dict(),
                "ema": {f"{hl:g}": ema.state_dict() for hl, ema in zip(halflives, emas)}}
        atomic_save(snap, run_dir / "snapshots" / f"kimg_{kimg:06d}.pt")
        print(f"[snapshot] kimg {kimg}")

    def save_latest():
        atomic_save({"net": net.state_dict(), "opt": opt.state_dict(), "emas": [e.state_dict() for e in emas],
                     "gen": gen.get_state(), "state": state, "args": vars(args)}, latest_path)

    def crossed(prev_nimg, nimg, every_kimg):
        every = every_kimg * 1000
        return every > 0 and math.floor(nimg / every) > math.floor(prev_nimg / every)

    total_nimg = args.total_kimg * 1000
    grid_steps = set(GRID_STEPS)
    if state["step"] == 0:
        save_grids()
    last_latest = time.time()
    wall_start = time.time() - state["wall_sec"]
    interval = {"loss": 0.0, "bins": torch.zeros(4), "counts": torch.zeros(4), "gn": 0.0, "n": 0,
                "t0": time.time(), "img": 0, "sec": 0.0}
    bad_steps = 0
    last_eval = {"fid": None, "fid_nimg": None, "snapshot_nimg": None}
    net.train()

    while state["nimg"] < total_nimg:
        if args.time_budget_h is not None and state["train_sec"] >= args.time_budget_h * 3600:
            print("Time budget reached")
            break
        step_t0 = time.time()
        lr = args.lr * min(1.0, (state["step"] + 1) / max(args.warmup_steps, 1))
        for group in opt.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=device)
        bin_sum = torch.zeros(4, device=device)
        bin_cnt = torch.zeros(4, device=device)
        for _ in range(accum):
            if pairs is not None:
                # Reflow: fixed (noise, image) pairs; no augmentation (it would break the pairing) or label dropout.
                noise, x0, y = pairs.batch(args.micro_batch)
            else:
                x0, y = data.batch(args.micro_batch, gen)
                x0, aug = augment(x0, args.p_aug, gen)
                drop = torch.rand(y.shape[0], device=device, generator=gen) < args.p_uncond
                y = torch.where(drop, torch.full_like(y, NULL_CLASS), y)
            if args.t_dist == "ushape":
                t = sample_t_ushape(y.shape[0], device, gen, args.t_a)
            else:
                t = sample_t(y.shape[0], device, gen, args.t_mean, args.t_std, args.p_uniform)
            if pairs is None:
                noise = torch.randn(x0.shape, device=device, generator=gen)
            x0 = x0.contiguous(memory_format=torch.channels_last)
            noise = noise.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if pairs is not None:
                    per_sample, mse = reflow_loss(cnet, x0, noise, t, y, args.loss)
                else:
                    per_sample, mse = fm_loss(cnet, x0, t, noise, y, aug, args.sigma_data)
            loss = per_sample.mean()
            (loss / accum).backward()
            loss_sum += loss.detach() / accum
            b = (t * 4).long().clamp(max=3)
            bin_sum.index_add_(0, b, mse.detach())
            bin_cnt.index_add_(0, b, torch.ones_like(mse))

        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            print(f"Non-finite gradient at step {state['step']} ({bad_steps} in a row)")
            if bad_steps >= 10:
                save_latest()
                raise SystemExit("Too many non-finite steps; aborting")
            continue
        bad_steps = 0
        opt.step()
        opt.zero_grad(set_to_none=True)
        prev_nimg = state["nimg"]
        state["nimg"] += args.batch
        state["step"] += 1
        for ema in emas:
            ema.update(net, state["nimg"], args.batch)

        step_sec = time.time() - step_t0
        state["train_sec"] += step_sec
        interval["loss"] += loss_sum.item()
        interval["bins"] += bin_sum.cpu()
        interval["counts"] += bin_cnt.cpu()
        interval["gn"] += grad_norm.item()
        interval["n"] += 1
        interval["img"] += args.batch
        interval["sec"] += step_sec

        if state["step"] % args.log_every == 0:
            state["wall_sec"] = time.time() - wall_start
            n = interval["n"]
            bins = (interval["bins"] / interval["counts"].clamp(min=1)).tolist()
            peak = torch.cuda.max_memory_allocated()
            row = {"step": state["step"], "kimg": f"{state['nimg'] / 1000:.1f}",
                   "epoch": f"{state['nimg'] / epoch_size:.2f}", "loss": f"{interval['loss'] / n:.5f}",
                   "mse_t0": f"{bins[0]:.5f}", "mse_t1": f"{bins[1]:.5f}", "mse_t2": f"{bins[2]:.5f}",
                   "mse_t3": f"{bins[3]:.5f}", "grad_norm": f"{interval['gn'] / n:.4f}", "lr": f"{lr:.2e}",
                   "img_per_sec": f"{interval['img'] / max(interval['sec'], 1e-9):.1f}",
                   "peak_mem_gib": f"{peak / 2 ** 30:.2f}", "train_h": f"{state['train_sec'] / 3600:.3f}",
                   "wall_h": f"{state['wall_sec'] / 3600:.3f}"}
            train_log.write(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            if peak > VRAM_LIMIT_BYTES:
                print(f"WARNING: peak memory {peak / 1e9:.2f} GB exceeds the 20 GB limit")
            interval = {"loss": 0.0, "bins": torch.zeros(4), "counts": torch.zeros(4), "gn": 0.0, "n": 0,
                        "t0": time.time(), "img": 0, "sec": 0.0}

        if state["step"] in grid_steps:
            save_grids()
        if crossed(prev_nimg, state["nimg"], args.fid_every_kimg):
            last_eval["fid"] = run_fid()
            last_eval["fid_nimg"] = state["nimg"]
        if crossed(prev_nimg, state["nimg"], args.snapshot_every_kimg):
            save_grids()
            save_snapshot()
            last_eval["snapshot_nimg"] = state["nimg"]
        if time.time() - last_latest > args.latest_every_min * 60:
            state["wall_sec"] = time.time() - wall_start
            save_latest()
            last_latest = time.time()

    # Final evaluation / snapshot unless the last step already produced them.
    state["wall_sec"] = time.time() - wall_start
    if last_eval["fid_nimg"] != state["nimg"]:
        save_grids()
        last_eval["fid"] = run_fid()
    if last_eval["snapshot_nimg"] != state["nimg"]:
        save_snapshot()
    final_fid = last_eval["fid"]
    save_latest()
    summary = {"step": state["step"], "kimg": state["nimg"] / 1000, "train_h": state["train_sec"] / 3600,
               "wall_h": (time.time() - wall_start) / 3600, "final_best_fid": final_fid,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "arch": args.arch,
               "num_params": num_params}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
