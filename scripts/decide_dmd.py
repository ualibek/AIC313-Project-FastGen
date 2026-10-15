"""Pick the main DMD2 config from the pilots (train-reference 1-NFE FID at equal training time).

Each pilot changes one setting of the base pilot P0. The main run takes P0 plus every change whose pilot beat
P0 by more than --margin FID at its final evaluation (best of raw / EMAs there), with ~0% memorization.
Prints the extra command-line flags for the main run; exits with code 2 if no pilot reached --max_fid.

    python scripts/decide_dmd.py --base runs/dmd_p0_base --pilot runs/dmd_p1_lr1e-5 "--lr_gen 1e-5 --lr_fake 1e-5" ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def final_best(run_dir):
    rows = list(csv.DictReader((Path(run_dir) / "fid.csv").open()))
    last = max(int(r["it"]) for r in rows)
    final = [r for r in rows if int(r["it"]) == last]
    best = min(final, key=lambda r: float(r["fid"]))
    return float(best["fid"]), float(best["memorized"]), best["ema_kimg"], float(best["kimg"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--pilot", nargs=2, action="append", default=[], metavar=("RUN_DIR", "FLAGS"))
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--max_fid", type=float, default=40.0)
    p.add_argument("--max_memorized", type=float, default=0.01)
    p.add_argument("--out", default="runs/dmd_pilot_decision.json")
    args = p.parse_args()

    base_fid, base_mem, base_key, base_kimg = final_best(args.base)
    results = {args.base: {"fid": base_fid, "memorized": base_mem, "ema": base_key, "kimg": base_kimg, "flags": ""}}
    flags = []
    for run_dir, run_flags in args.pilot:
        if not (Path(run_dir) / "fid.csv").exists():
            print(f"{run_dir}: no fid.csv, skipped", file=sys.stderr)
            continue
        fid, mem, key, kimg = final_best(run_dir)
        win = fid < base_fid - args.margin and mem <= args.max_memorized
        results[run_dir] = {"fid": fid, "memorized": mem, "ema": key, "kimg": kimg, "flags": run_flags, "adopted": win}
        if win:
            flags.append(run_flags)
    for run_dir, r in results.items():
        print(f"{run_dir}: FID {r['fid']:.2f} ({r['ema']}, {r['kimg']:.0f} kimg, memorized {r['memorized']:.2%})"
              f"{' -> adopted' if r.get('adopted') else ''}", file=sys.stderr)
    best = min(r["fid"] for r in results.values())
    decision = {"pilots": results, "main_flags": " ".join(flags), "best_fid": best, "proceed": best <= args.max_fid}
    Path(args.out).write_text(json.dumps(decision, indent=2))
    if best > args.max_fid:
        print(f"no pilot reached FID {args.max_fid}; stopping", file=sys.stderr)
        sys.exit(2)
    print(" ".join(flags))


if __name__ == "__main__":
    main()
