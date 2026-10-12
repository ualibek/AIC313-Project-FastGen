"""Sampler / guidance sweep for a flow-matching snapshot, scored by train-reference FID.

    python eval_fm.py --snapshot runs/fm_main/snapshots/kimg_030000.pt --ema 2000
Writes <run>/eval/<snapshot>_ema<key>.csv and a 64-image grid per setting.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from fastgen.data import balanced_classes, grid_classes, load_cache, to_eval_uint8
from fastgen.fid import FIDMonitor
from fastgen.flow import NFECounter, finalize, sample_euler, sample_heun
from fastgen.unet import FlowUNet

CFG_INTERVAL = (0.2, 0.85)  # guidance only at intermediate noise levels (Kynkaanniemi et al., 2024)


def sweep_configs(preset="full"):
    if preset == "pairs":
        # Candidate teacher samplers for generating reflow pairs (guided ODE, ~50-130 NFE).
        configs = [dict(sampler="euler", steps=s, shift=1.0, cfg=1.5, interval=CFG_INTERVAL) for s in (32, 64)]
        for steps in (16, 24, 32):
            for cfg in (1.25, 1.5, 2.0):
                configs.append(dict(sampler="heun", steps=steps, shift=1.0, cfg=cfg, interval=CFG_INTERVAL))
        configs.append(dict(sampler="heun", steps=40, shift=1.0, cfg=1.5, interval=CFG_INTERVAL))
        configs.append(dict(sampler="heun", steps=32, shift=1.0, cfg=1.5, interval=None))
        return configs
    if preset == "student":
        # <= 4 NFE samplers for a reflowed (2-rectified) student; no guidance (it is baked into the pairs).
        base = dict(shift=1.0, cfg=1.0, interval=None)
        configs = [dict(base, sampler="euler", steps=1)]
        configs += [dict(base, sampler="euler", steps=2, t_grid=(1.0, t2, 0.0)) for t2 in (0.5, 0.6, 0.7, 0.8, 0.9)]
        for steps in (3, 4):
            for shift in (0.5, 1.0, 2.0, 3.0):
                configs.append(dict(base, sampler="euler", steps=steps, shift=shift))
            for shift in (1.0, 1.5, 2.0, 3.0):
                configs.append(dict(base, sampler="euler", steps=steps, shift=shift, update="rfpp"))
        configs.append(dict(base, sampler="heun", steps=2))  # 3 NFE
        configs.append(dict(base, sampler="heun", steps=2, final_correction=True))  # 4 NFE
        return configs
    if preset == "cfg":
        configs = [dict(sampler="euler", steps=32, shift=1.0, cfg=1.0, interval=None)]
        for cfg in (1.25, 1.5, 2.0, 3.0, 4.0):
            for interval in (None, CFG_INTERVAL):
                configs.append(dict(sampler="euler", steps=32, shift=1.0, cfg=cfg, interval=interval))
        return configs
    quick = preset == "quick"
    configs = []
    for steps in (1, 2, 4, 8, 16, 32, 64):
        for shift in ((1.0,) if steps == 1 else (0.5, 1.0, 2.0)):
            if quick and shift != 1.0:
                continue
            configs.append(dict(sampler="euler", steps=steps, shift=shift, cfg=1.0, interval=None))
    if not quick:
        for steps in (4, 8, 16, 32):
            configs.append(dict(sampler="heun", steps=steps, shift=1.0, cfg=1.0, interval=None))
    for cfg in (1.25, 1.5, 2.0, 3.0, 4.0):
        for interval in (None, CFG_INTERVAL):
            configs.append(dict(sampler="euler", steps=32, shift=1.0, cfg=cfg, interval=interval))
    # <= 4 NFE budget with guidance: 2 guided steps (2 x 2 NFE)
    for cfg in (1.5, 2.0, 3.0):
        for shift in (0.5, 1.0, 2.0):
            configs.append(dict(sampler="euler", steps=2, shift=shift, cfg=cfg, interval=None))
    configs.append(dict(sampler="euler", steps=32, shift=1.0, cfg=1.0, interval=None, half_lsb=False))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--ema", default="2000")
    parser.add_argument("--preset", default="full", choices=["full", "quick", "cfg", "pairs", "student"])
    parser.add_argument("--fp32", action="store_true", help="sample in fp32 (default bf16 autocast)")
    parser.add_argument("--batch", type=int, default=302)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--fid_batch", type=int, default=128, help="Inception batch size")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    net = FlowUNet(sigma_data=snap["sigma_data"], **snap["unet_config"]).to(device).eval()
    weights = snap["raw"] if args.ema == "raw" else snap["ema"][args.ema]
    net.load_state_dict(weights)
    net.requires_grad_(False)
    net = net.to(memory_format=torch.channels_last)
    if not args.no_compile:
        net = torch.compile(net, dynamic=False)

    cache = load_cache()
    monitor = FIDMonitor(device, num_workers=4, batch_size=args.fid_batch)
    monitor.load_or_build_reference(cache)
    labels = balanced_classes().to(device)
    noise = torch.randn(len(labels), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(1234))
    grid_y = grid_classes().to(device)
    grid_noise = torch.randn(len(grid_y), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(123))

    out_dir = Path(args.snapshot).parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    tag = f"{Path(args.snapshot).stem}_ema{args.ema}_{args.preset}"
    fields = ["sampler", "steps", "nfe", "shift", "cfg", "interval", "t_grid", "update", "half_lsb", "fid", "seconds"]
    with (out_dir / f"{tag}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cfg in sweep_configs(args.preset):
            t0 = time.time()
            fn = sample_euler if cfg["sampler"] == "euler" else sample_heun
            half_lsb = cfg.get("half_lsb", True)
            kwargs = dict(shift=cfg["shift"], cfg_scale=cfg["cfg"], cfg_interval=cfg["interval"])
            kwargs.update({k: cfg[k] for k in ("t_grid", "update", "final_correction") if k in cfg})
            counter = NFECounter(net, args.batch)
            outs = []
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                for i in range(0, len(labels), args.batch):
                    counter.batch_size = len(labels[i:i + args.batch])
                    x = fn(counter, noise[i:i + args.batch], labels[i:i + args.batch], cfg["steps"], **kwargs)
                    outs.append(finalize(x.float(), half_lsb))
                grid = fn(net, grid_noise, grid_y, cfg["steps"], **kwargs)
            nfe = counter.nfe / -(-len(labels) // args.batch)
            fid = monitor.fid(to_eval_uint8(torch.cat(outs)))
            row = dict(sampler=cfg["sampler"], steps=cfg["steps"], nfe=f"{nfe:g}", shift=cfg["shift"], cfg=cfg["cfg"],
                       interval=cfg["interval"], t_grid=cfg.get("t_grid"),
                       update=cfg.get("update", "euler") + ("+fc" if cfg.get("final_correction") else ""),
                       half_lsb=half_lsb, fid=f"{fid:.3f}", seconds=f"{time.time() - t0:.0f}")
            writer.writerow(row)
            f.flush()
            print(row, flush=True)
            name = f"{tag}_{cfg['sampler']}{cfg['steps']}_s{cfg['shift']}_w{cfg['cfg']}_{'int' if cfg['interval'] else 'full'}"
            if "t_grid" in cfg:
                name += "_t" + "-".join(f"{t:g}" for t in cfg["t_grid"])
            if cfg.get("update", "euler") != "euler" or cfg.get("final_correction"):
                name += f"_{cfg.get('update', 'euler')}{'_fc' if cfg.get('final_correction') else ''}"
            save_image(to_eval_uint8(finalize(grid.float(), half_lsb)).float() / 255, out_dir / f"{name}.png", nrow=8)


if __name__ == "__main__":
    main()
