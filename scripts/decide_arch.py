"""Pick the teacher architecture from the two equal-time pilot runs.

Rule (fixed before seeing results): keep unet79 (faster, more parameter headroom) unless unet68's
final train-reference FID (Euler-32, EMA) is better by more than 3%. A crashed pilot loses.
Writes the decision to runs/pilot_decision.json and prints the chosen arch on the last line.
"""

import csv
import json
import math
from pathlib import Path

RUNS = {"unet79": Path("runs/pilot_unet79"), "unet68": Path("runs/pilot_unet68")}


def curve(run_dir):
    path = run_dir / "fid.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [(float(r["kimg"]), float(r["train_h"]), float(r["fid"]))
                for r in csv.DictReader(f) if r["sampler"] == "euler32"]


def main():
    curves = {arch: curve(d) for arch, d in RUNS.items()}
    final = {arch: (c[-1][2] if c else math.inf) for arch, c in curves.items()}
    choice = "unet68" if final["unet68"] < 0.97 * final["unet79"] else "unet79"
    if math.isinf(final["unet79"]) and math.isinf(final["unet68"]):
        choice = "none"
    decision = {"choice": choice, "final_fid_euler32": final, "curves (kimg, train_h, fid)": curves,
                "rule": "unet68 only if FID_68 < 0.97 * FID_79 at equal training time"}
    Path("runs/pilot_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print(choice)


if __name__ == "__main__":
    main()
