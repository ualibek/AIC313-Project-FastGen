"""Export a training snapshot's weights as evaluate.py-compatible checkpoints.

    python export_ckpt.py --snapshot runs/<run>/snapshots/kimg_XXXXXX.pt --ema 2000 --modes one_nfe few_nfe
"""

import argparse
from pathlib import Path

import torch

import model as model_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--ema", default="2000", help="EMA half-life key (kimg) or 'raw'")
    parser.add_argument("--modes", nargs="+", default=["one_nfe", "few_nfe"], choices=["one_nfe", "few_nfe"])
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    weights = snap["raw"] if args.ema == "raw" else snap["ema"][args.ema]
    weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
    classes = {"one_nfe": (model_lib.ModelOneNFE, model_lib.ONE_NFE_BACKBONE),
               "few_nfe": (model_lib.ModelFewNFE, model_lib.FEW_NFE_BACKBONE)}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        cls, backbone_cfg = classes[mode]
        expected = dict(snap["unet_config"], sigma_data=snap["sigma_data"])
        if {k: backbone_cfg[k] for k in expected} != {k: (tuple(v) if isinstance(v, list) else v)
                                                       for k, v in expected.items()}:
            raise SystemExit(f"{mode}: model.py backbone config {backbone_cfg} != snapshot config {expected}")
        model = cls()
        model.backbone.load_state_dict(weights)
        out = Path(args.out_dir) / f"{mode}.ckpt"
        torch.save({"state_dict": model.state_dict()}, out)
        print(f"{mode}: wrote {out} ({model.count_parameters():,} params) from {args.snapshot} [ema={args.ema}]")


if __name__ == "__main__":
    main()
