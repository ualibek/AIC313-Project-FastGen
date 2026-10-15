"""Training entry point.

Stages:
    python train.py fm --run_dir runs/<name> [options]          # flow-matching teacher (fastgen/train_fm.py)
    python train.py pairs --snapshot <teacher> --out pairs/<name> [options]   # reflow pairs (fastgen/pairs.py)
    python train.py reflow --run_dir runs/<name> --pairs pairs/<name> --init_snapshot <teacher> [options]
        # 2-rectified flow++ (Lee et al., 2024): the fm trainer with the reflow defaults below
    python train.py dmd --run_dir runs/<name> [options]
        # DMD2 (Yin et al., 2024) from the 2-RF student and the guided teacher (fastgen/train_dmd.py)
"""

import sys

# Reflow defaults; flags given on the command line come later and override them.
REFLOW_DEFAULTS = [
    "--t_dist", "ushape", "--t_a", "4", "--loss", "huber", "--p_uncond", "0", "--p_aug", "0",
    "--batch", "256", "--micro_batch", "64", "--lr", "2e-4", "--warmup_steps", "5000",
    "--ema_halflives_kimg", "250,1000,2000", "--total_kimg", "40000", "--snapshot_every_kimg", "4000",
    "--fid_every_kimg", "1000", "--fid_sampler", "1", "--fid_extra", "2,2@0.8,4", "--grid_samplers", "1,2,4",
]
STAGES = {"fm", "pairs", "reflow", "dmd"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: python train.py {{{','.join(sorted(STAGES))}}} [options]")
    stage, argv = sys.argv[1], sys.argv[2:]
    if stage == "fm":
        from fastgen.train_fm import main as train_fm
        train_fm(argv)
    elif stage == "reflow":
        from fastgen.train_fm import main as train_fm
        train_fm(REFLOW_DEFAULTS + argv)
    elif stage == "dmd":
        from fastgen.train_dmd import main as train_dmd
        train_dmd(argv)
    elif stage == "pairs":
        from fastgen.pairs import main as make_pairs
        make_pairs(argv)


if __name__ == "__main__":
    main()
