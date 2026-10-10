"""Training entry point.

Stages:
    python train.py fm --run_dir runs/<name> [options]   # flow-matching teacher (see fastgen/train_fm.py)
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"fm"}:
        raise SystemExit("usage: python train.py fm --run_dir runs/<name> [options]")
    stage, argv = sys.argv[1], sys.argv[2:]
    if stage == "fm":
        from fastgen.train_fm import main as train_fm
        train_fm(argv)


if __name__ == "__main__":
    main()
