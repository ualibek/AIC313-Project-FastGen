"""Reflow loss as a function of t (the analogue of Lee et al., 2024, Fig. 3) for one or more snapshots.

    python scripts/reflow_loss_curve.py --pairs pairs/<name> --snapshots A.pt B.pt --ema 1000 --out curve.csv
Velocity MSE of each model on the same 4,096 pairs (the last 64 groups) at 20 evenly spaced t, with fixed noise.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastgen.data import PairData, group_noise  # noqa: E402
from fastgen.pairs import load_teacher  # noqa: E402

NUM_T = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument("--ema", default="1000", help="EMA key; snapshots lacking it fall back to their largest EMA")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    data = PairData(args.pairs, device)
    assert not data.stored_z, "use a seeded (synthetic) pair set"
    groups = list(range(data.units - args.num_groups, data.units))
    ts = [(i + 0.5) / NUM_T for i in range(NUM_T)]
    rows = []
    for path in args.snapshots:
        keys = torch.load(path, map_location="cpu", weights_only=False)["ema"].keys()
        ema = args.ema if args.ema in keys else max(keys, key=float)
        net = load_teacher(path, ema, device, compile_net=False)
        sums = torch.zeros(NUM_T)
        noise_gen = torch.Generator(device=device)
        for g in groups:
            sl = slice(g * data.group, (g + 1) * data.group)
            z = group_noise(data.base_seed, g, data.group, device, noise_gen)
            x0 = data.x[sl].to(device).float() / 127.5 - 1
            y = data.y[sl].to(device)
            for i, t in enumerate(ts):
                tt = torch.full((len(y),), t, device=device)
                xt = (1 - t) * x0 + t * z
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    v = net(xt, tt, class_labels=y)
                sums[i] += (v.float() - (z - x0)).pow(2).mean().item()
        for t, s in zip(ts, sums / len(groups)):
            rows.append({"snapshot": path, "ema": ema, "t": f"{t:.3f}", "v_mse": f"{s:.5f}"})
        print(path, ema, " ".join(f"{s:.4f}" for s in (sums / len(groups)).tolist()), flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snapshot", "ema", "t", "v_mse"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
