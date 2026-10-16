"""Convert a DMD latest.pt (pilot run with --snapshot_every_kimg 0) into the train_fm/train_dmd snapshot format
(arch, sigma_data, unet_config, raw, ema, dmd), so export_ckpt.py can export it. Same fields as save_snapshot().
    python scripts/latest_to_snapshot.py runs/<run>/latest.pt runs/<run>/snapshots/kimg_XXXXXX.pt"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastgen.flow import time_grid
from fastgen.unet import UNET_CONFIGS
src, dst = sys.argv[1:3]
ck = torch.load(src, map_location="cpu", weights_only=False)
a, st = ck["args"], ck["state"]
halflives = [float(h) for h in a["ema_halflives_kimg"].split(",")]
snap = {"arch": a["arch"], "sigma_data": a["sigma_data"], "unet_config": UNET_CONFIGS[a["arch"]],
        "nimg": st["nimg"], "step": st["gstep"], "raw": ck["gen"],
        "ema": {f"{hl:g}": sd for hl, sd in zip(halflives, ck["emas"])},
        "dmd": {"steps": a["steps"], "shift": a["shift"], "update": a.get("update", "rfpp"),
                "time_grid": time_grid(a["steps"], a["shift"]).tolist(), "converted_from": src}}
torch.save(snap, dst)
print(f"{src} ({st['nimg'] / 1000:.0f} kimg) -> {dst}; ema keys {list(snap['ema'])}")
