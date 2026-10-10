"""Train-split cache and GPU-resident batch sampler with augmentation."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PokemonDataModule

DEFAULT_CACHE = Path("cache/train64.pt")


def build_cache(
    out_path=DEFAULT_CACHE,
    data_root="./data/pokemon-generation-one-22k",
    split_dir="./data/pokemon-generation-one-22k",
    num_workers=8,
):
    """Decode + resize the train split once with the provided dataset pipeline; store uint8."""
    module = PokemonDataModule(
        data_root=data_root, split_dir=split_dir, batch_size=256,
        num_workers=num_workers, return_category=True,
    )
    dataset = module.train_dataset
    # Own loader: module.train_dataloader() uses drop_last=True and shuffles.
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers, drop_last=False)
    images, labels = [], []
    for x, y in loader:
        # Exact inverse of ToTensor + Normalize((0.5,) * 3, (0.5,) * 3).
        images.append(((x + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8))
        labels.append(y)
    images = torch.cat(images)
    labels = torch.cat(labels)
    xf = images.float() / 127.5 - 1
    stats = {
        "mean": xf.mean().item(),
        "std": xf.std().item(),
        "rms": xf.pow(2).mean().sqrt().item(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels, "paths": list(dataset.paths), "stats": stats}, out_path)
    print(f"Cached {len(images)} images to {out_path}; stats {stats}")
    return out_path


def load_cache(path=DEFAULT_CACHE):
    path = Path(path)
    if not path.exists():
        build_cache(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def reference_uint8(images_uint8):
    """Reproduce evaluate.prepare_reference_set's pixels from cached uint8 images.

    The reference path is ToTensor -> Normalize -> (x + 1) / 2 -> clamp -> to_pil_image, and
    to_pil_image truncates (`mul(255).byte()`), which shifts some levels down by one.
    """
    x = images_uint8.float().div(255)
    x = x.sub(0.5).div(0.5)
    x = ((x + 1) / 2).clamp(0, 1)
    return x.mul(255).byte()


def to_eval_uint8(samples):
    """Quantize model outputs exactly like evaluate.generate_samples (truncating byte cast)."""
    return ((samples.float() + 1) / 2).clamp(0, 1).mul(255).byte()


class GPUData:
    """Whole train split as a uint8 tensor on the GPU; epoch-wise shuffled batches."""

    def __init__(self, cache, device):
        self.images = cache["images"].to(device)
        self.labels = cache["labels"].to(device)
        self.num = self.images.shape[0]
        self.device = device
        self.perm = None
        self.pos = 0

    def batch(self, size, generator):
        out = []
        while size > 0:
            if self.perm is None or self.pos >= self.num:
                self.perm = torch.randperm(self.num, device=self.device, generator=generator)
                self.pos = 0
            take = min(size, self.num - self.pos)
            out.append(self.perm[self.pos:self.pos + take])
            self.pos += take
            size -= take
        idx = torch.cat(out)
        x = self.images[idx].float() / 127.5 - 1
        return x, self.labels[idx]


def augment(x, p_aug, generator, max_translate=8, scale_std=0.2):
    """Random h-flip (unconditioned) + non-leaking geometric augmentation (EDM-style).

    Integer translation and isotropic scaling are each applied with probability p_aug; their
    parameters are returned as augmentation labels that condition the network (zeros = no
    augmentation, which is what we use at sampling time).
    """
    b, _, h, w = x.shape
    dev = x.device
    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    labels = torch.zeros(b, 3, device=dev)
    if p_aug <= 0:
        return x, labels

    do_t = torch.rand(b, device=dev, generator=generator) < p_aug
    do_s = torch.rand(b, device=dev, generator=generator) < p_aug
    shift = torch.randint(-max_translate, max_translate + 1, (b, 2), device=dev, generator=generator).float()
    shift = shift * do_t[:, None]
    log2s = torch.randn(b, device=dev, generator=generator) * scale_std * do_s
    scale = torch.exp2(log2s)
    # Output pixel p samples input at (p - shift) / scale (normalized coordinates, align_corners=False).
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = 1 / scale
    theta[:, 1, 1] = 1 / scale
    theta[:, 0, 2] = -shift[:, 0] * (2 / w) / scale
    theta[:, 1, 2] = -shift[:, 1] * (2 / h) / scale
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    applied = (do_t | do_s)[:, None, None, None]
    x = torch.where(applied, warped, x)
    labels = torch.stack([shift[:, 0] / max_translate, shift[:, 1] / max_translate, log2s / scale_std], dim=1)
    return x, labels


def balanced_classes(num_classes=151, per_class=20):
    """Class layout used by evaluate.py: each class id repeated per_class times."""
    return torch.arange(num_classes, dtype=torch.long).repeat_interleave(per_class)


def grid_classes(num=64, num_classes=151):
    """Evenly spaced class ids for fixed sample grids."""
    return torch.tensor([math.floor(i * num_classes / num) for i in range(num)], dtype=torch.long)
