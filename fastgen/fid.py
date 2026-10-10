"""In-process FID on the clean-fid pipeline used by evaluate.py, against a train-split reference.

Images enter as uint8 [N, 3, 64, 64] tensors already quantized the way evaluate.py quantizes
(see fastgen.data.to_eval_uint8 / reference_uint8); they then follow clean-fid's "clean" mode
exactly: PIL bicubic resize to 299x299 without re-quantization, then the torchscript Inception.
"""

import json
from pathlib import Path

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance
from cleanfid.resize import build_resizer
from torch.utils.data import DataLoader, Dataset

from fastgen.data import reference_uint8

DEFAULT_REF = Path("cache/fid_ref_train.npz")


class _ResizeArray(Dataset):
    def __init__(self, images_nhwc):
        self.images = images_nhwc
        self.fn_resize = build_resizer("clean")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        resized = self.fn_resize(self.images[i])  # float32 HWC in [0, 255]
        return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))


class FIDMonitor:
    def __init__(self, device, num_workers=8, batch_size=128):
        self.device = torch.device(device)
        self.model = build_feature_extractor("clean", self.device, use_dataparallel=False)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.ref_mu = None
        self.ref_sigma = None

    @torch.no_grad()
    def features(self, images_uint8):
        arr = images_uint8.cpu().numpy().transpose(0, 2, 3, 1)
        loader = DataLoader(_ResizeArray(arr), batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, drop_last=False)
        feats = [self.model(batch.to(self.device)).cpu().numpy() for batch in loader]
        return np.concatenate(feats).astype(np.float64)

    def load_or_build_reference(self, cache, ref_path=DEFAULT_REF, info_path=None):
        """Class-balanced (weight 1/n_c per image) Inception stats of the whole train split."""
        ref_path = Path(ref_path)
        if ref_path.exists():
            ref = np.load(ref_path)
            self.ref_mu, self.ref_sigma = ref["mu"], ref["sigma"]
            return
        labels = cache["labels"].numpy()
        feats = self.features(reference_uint8(cache["images"]))
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        self.ref_mu = np.average(feats, axis=0, weights=weights)
        self.ref_sigma = np.cov(feats, rowvar=False, aweights=weights)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref_path, mu=self.ref_mu, sigma=self.ref_sigma)

        # Diagnostics: FID of a balanced 20-per-class subset of the train images themselves
        # (a perfect copy of the training set would score about this).
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), 20, replace=False)
                              for c in range(len(counts))])
        copy_floor = self.fid_from_feats(feats[idx])
        info = {"num_images": int(len(labels)), "copy_floor_fid": float(copy_floor)}
        info_path = Path(info_path or ref_path.with_suffix(".json"))
        info_path.write_text(json.dumps(info, indent=2))
        print(f"Built FID reference: {info}")

    def fid_from_feats(self, feats):
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        return float(frechet_distance(mu, sigma, self.ref_mu, self.ref_sigma))

    def fid(self, images_uint8):
        return self.fid_from_feats(self.features(images_uint8))


@torch.no_grad()
def memorization_ratio(samples, train_images_uint8, threshold=1 / 3, chunk=2048):
    """Fraction of samples whose nearest / second-nearest train image distance ratio < threshold.

    Pixel-space L2 (Gu et al., 2023). Each train image counts once, as the closer of itself
    and its horizontal flip (training uses flips).
    """
    x = samples.float().flatten(1)
    best = torch.full((x.shape[0], 2), float("inf"), device=x.device)
    x_sq = x.pow(2).sum(1, keepdim=True)
    for start in range(0, train_images_uint8.shape[0], chunk):
        imgs = train_images_uint8[start:start + chunk].to(x.device).float() / 127.5 - 1
        dists = []
        for view in (imgs, imgs.flip(3)):
            y = view.flatten(1)
            dists.append(x_sq + y.pow(2).sum(1)[None] - 2 * x @ y.T)
        d = torch.minimum(dists[0], dists[1]).clamp_min(0)
        best = torch.cat([best, d], dim=1).topk(2, dim=1, largest=False).values
    ratio = (best[:, 0] / best[:, 1].clamp_min(1e-12)).sqrt()
    return float((ratio < threshold).float().mean())
