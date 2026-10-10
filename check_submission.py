"""Pre-submission checks for both checkpoints, using evaluate.py's own loading and sampling code.

Checks: loads via Model.load_checkpoint, <= 100M parameters, NFE budget counted by a hook on the
backbone (rows processed / batch size, so a batched CFG pair counts as 2) while running
evaluate.generate_samples (batch 32 with a partial last batch), finite outputs of the right shape.
"""

import argparse
import shutil
import tempfile

import torch

import evaluate
from model import Model

BUDGETS = {"one_nfe": (1, 1), "few_nfe": (1, 4)}


def check(mode, path, device):
    model = Model.load_checkpoint(path, evaluate_mode=mode, device=device)
    params = model.count_parameters()
    assert params <= evaluate.MAX_MODEL_PARAMETERS, f"{mode}: {params:,} parameters"

    rows = {"n": 0}
    model.backbone.register_forward_pre_hook(lambda module, args: rows.__setitem__("n", rows["n"] + args[0].shape[0]))
    nfes = []
    original_sample = model.sample

    def counted_sample(shape, **kwargs):
        rows["n"] = 0
        out = original_sample(shape, **kwargs)
        nfes.append(rows["n"] / shape[0])
        assert out.shape == tuple(shape), f"{mode}: output shape {tuple(out.shape)} != {tuple(shape)}"
        assert torch.isfinite(out).all(), f"{mode}: non-finite output"
        return out

    model.sample = counted_sample
    categories = torch.arange(151).repeat_interleave(20)[:76]  # 32 + 32 + 12 (partial batch)
    tmp = tempfile.mkdtemp()
    try:
        evaluate.generate_samples(model, tmp, categories, batch_size=32, device=device)
    finally:
        shutil.rmtree(tmp)
    lo, hi = BUDGETS[mode]
    assert all(lo <= n <= hi for n in nfes), f"{mode}: NFE per sample() call {nfes} outside [{lo}, {hi}]"
    print(f"{mode}: OK  params={params:,}  NFE per call={sorted(set(nfes))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one_nfe", default="./checkpoints/one_nfe.ckpt")
    parser.add_argument("--few_nfe", default="./checkpoints/few_nfe.ckpt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    check("one_nfe", args.one_nfe, args.device)
    check("few_nfe", args.few_nfe, args.device)


if __name__ == "__main__":
    main()
