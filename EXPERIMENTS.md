# Experiment log and model lineage

This file records every training run behind our submission: what each model was trained from, which network
teaches which, how the runs differ, and the command path to reproduce the numbers we report. It is updated as
runs finish. Papers and code we build on are listed in `CITATIONS.md`.

## Conventions

- **FID** is clean-fid FID against the class-balanced **training** split: 3,020 samples (20 per class), noise seed
  1234 (`fastgen/fid.py`). Every model and hyperparameter choice below is made on this number.
- The **validation split** is used only for one-off sanity checks with the official `evaluate.py`, never for
  selection. Val FIDs run about 10–25 higher than train-ref FIDs; 3,020 real train images score ~21 on val.
- **kimg / Mimg** are thousands / millions of images seen by the network being trained (for DMD: generator images).
- **EMA k** is an exponential moving average of the weights with a half-life of k kimg.
- **Architecture:** every network is `unet79` (78.9M parameters), trained from scratch on the training split only.
- **Determinism:** runs use fixed seeds (`--seed 0`). bf16 autocast, TF32, `cudnn.benchmark` and `torch.compile`
  make GPU runs non-bitwise-reproducible, so a re-run should land within FID noise (about ±0.3 at 3k samples).

## Lineage

```
train split (17,079 images, 151 classes)
│
└─ runs/fm_main ─ Stage 1 TEACHER: class-conditional flow matching, 30 Mimg
   │  (the teacher everywhere below = snapshots/kimg_030000.pt, EMA 2000)
   │
   ├─ pairs/fm30k_euler64_w1.5 ─ 1.024M (noise, image) pairs from the guided teacher ODE
   │  └─ runs/rf2_main ─ Stage 2 STUDENT: 2-rectified flow++ on those pairs, init = teacher weights, 60 Mimg
   │     │  kimg_060000 EMA 2000  <- shipped now in checkpoints/{one,few}_nfe.ckpt
   │     ├─ runs/rf2_real  (fine-tune on inverted real pairs; failed)
   │     │
   │     └─ Stage 3 DMD2, Phase A (1-step). Generator init = student; real score = teacher (frozen);
   │        │                                fake score init = teacher, plus a new GAN head
   │        ├─ runs/dmd_p0_base, dmd_p1_lr1e-5, dmd_p3_w1  (pilots)
   │        ├─ runs/dmd1_main -> best.pt 18.03 -> runs/dmd1b_lr2e-6  (restart; failed)
   │        ├─ runs/dmd_p2_ttur2 -> continued as runs/dmd1c_p2cont
   │        │  │  best.pt EMA 500 = 17.50  <- best 1-step model so far
   │        │  └─ Phase B v2+ (4-step). Generator init = dmd1c best.pt EMA 500; real score = teacher;
   │        │     fake score init = teacher
   │        │     dmd4b, dmd4c, dmd4f, dmd4h, dmd4i (current best)
   │        └─ runs/dmd4_main  Phase B v1 (4-step), generator init = student -> best.pt 15.46
```

## Who teaches whom

| Network | Learns from | Initialized from | Shipped? |
|---|---|---|---|
| Teacher (`fm_main`) | Train images: flow-matching loss | random | no |
| 2-RF++ student (`rf2_main`) | The teacher: regression onto 1.024M guided-teacher ODE pairs | teacher weights | yes, currently (1 and 4 NFE) |
| DMD generator | **DM term:** the frozen teacher (real score) against the fake score. **GAN term:** train images vs. its own samples. | Phase A: the student. Phase B v2+: the Phase A generator. | candidate |
| DMD fake score + GAN head (83.6M) | **Fake score:** flow matching on the generator's samples. **GAN head:** discriminator loss, train images vs. generator samples. | fake score: teacher weights; head: random | no (training only) |

**There is one teacher: `fm_main`.** Every later model is *initialized* from the previous stage (student ← teacher
weights, Phase A generator ← student, Phase B generator ← Phase A generator), but initialization is not supervision.
All supervision comes from `fm_main` (directly as the frozen real score, or through its ODE pairs for the student)
plus the training images (through the GAN term).

Every learned network stays ≤ 100M parameters individually (README, "Parameter limit"): teacher 78.9M, generator
78.9M, fake score + GAN head 83.6M (head 4.72M). Only the generator is in the submission.

## Stages 1–2: teacher and 2-RF++ student

| Run | What | Key settings | Length | Train-ref FID | Notes |
|---|---|---|---|---|---|
| `pilot_unet68` / `pilot_unet79` | Architecture pilots, equal wall-clock | flow matching, lr 2e-4 | 2 h each | 36.29 / 36.43 (Euler-32) | Tied; unet79 kept |
| `fm_main` | Teacher | lr 2e-4, batch 128, p_uncond 0.1 (for CFG), p_aug 0.15, logit-normal t (−0.8, 1.6) + 10% uniform, EMA 500/2000/5000 | 30 Mimg, 23.5 h | Euler-32 16.56; guided Euler-64 (w = 1.5 on t ∈ [0.2, 0.85]) 13.54 | Val: 30.30 / 21.81 |
| `pairs/fm30k_euler64_w1.5` | Teacher ODE pairs | guided Euler-64, w = 1.5 on t ∈ [0.2, 0.85]; 20 shards × 800 groups × 64 | ~16.5 h | — | Stores uint8 images only; noise is regenerated from per-group seeds (base seed 1e7) |
| `rf2_main` | 2-RF++ student | batch 256, lr 2e-4, U-shaped t (a = 4), pseudo-Huber loss, no CFG, no augmentation, EMA 250/1000/2000 | 60 Mimg, 43.9 h | 1-NFE 47.49; 4-NFE (RF++ update, shift 2) 16.28 | **Shipped.** Val: 71.45 / 27.27 |
| `rf2_real` | Fine-tune on inverted real images | lr 1e-5, init `rf2_main` 40 Mimg | stopped at 563 kimg | 1-NFE 51.3 → 85.6 (256 kimg) → 83.5 (512 kimg) | Failed: inverted noise is not white (pushes high-frequency texture into the noise) |

## Stage 3: DMD2 (`python train.py dmd`; code in `fastgen/train_dmd.py`, `fastgen/dmd.py`)

**Common settings** unless a row says otherwise:
- **Networks:** real score = teacher (EMA 2000, frozen); fake score = fresh copy of the teacher, with 500 fake-only
  warm-up iterations; GAN head on the fake score's 8×8 bottleneck.
- **Optimization:** batch 128 (micro-batch 64); AdamW (0.9, 0.999), weight decay 0.01; gradient clip 10; 500
  linear lr warm-up iterations.
- **Losses:** GAN weights D 1e-2 / G 3e-3; DM timesteps Karras t ∈ [0.003, 0.986].
- **Bookkeeping:** generator EMAs 50/200/500 (no ramp-up); seed 0.
- **TTUR** = fake-score updates per generator update.

### Phase A: 1-step generator (`ModelOneNFE`); FID = 1-NFE train-ref

| Run | Generator init | Differs from common | Length | Best FID | Outcome |
|---|---|---|---|---|---|
| `dmd_p0_base` | student | lr 2e-6, TTUR 5, real score guided w = 1.5 on [0.2, 0.85] | 322 kimg (2.5 h) | 20.18 | Baseline pilot (student 47.49) |
| `dmd_p1_lr1e-5` | student | P0 + lr 1e-5 (generator and fake) | 322 kimg | 18.46 | Fastest, but raw FID drifted up after 150 kimg |
| `dmd_p2_ttur2` | student | P0 + TTUR 2 | 639 kimg | 19.60 | Same per-image quality as TTUR 5 at 2× throughput |
| `dmd_p3_w1` | student | P0 + cfg 1.0 (unguided real score) | 328 kimg | 19.32 | Unguided beats guided |
| `dmd1_main` | student | lr 1e-5, TTUR 2, cfg 1.0 (all three pilot changes) | 1,250 kimg (4.6 h), stopped | 18.03 (EMA 200 @ 750) | Drifted after 750 kimg (raw 18.6 → 22.1). Exported to `checkpoints/dmd1_kimg000750_ema200/` (val 28.35) |
| `dmd1b_lr2e-6` | `dmd1_main` best.pt EMA 200 | lr 2e-6; fake score + head inherited from `dmd1_main` | 500 kimg, stopped | no gain | Failed: the inherited, drifted discriminator carries the drift |
| `dmd1c_p2cont` | student (it *is* the P2 run) | 0–639 kimg as P2 (cfg 1.5); 639–3,250 kimg continued with cfg 1.0, FID every 250 | 3,250 kimg (11.9 h incl. P2), stopped | **17.50 (EMA 500 @ 2,750)** | Best 1-step model. Raw FID slowly rose after ~2.5 Mimg (D logits +0.48/−0.35) |

### Phase B: 4-step generator (`ModelFewNFE`); FID = 4-NFE train-ref

Two time grids appear:
- **v1:** RF++ update `x ← (1−t′)·x̂ + t′·z` at shift 2, t = (1, .857, .667, .4, 0). This matches the shipped
  `FEW_NFE_SAMPLER`.
- **v2+:** plain Euler `x ← x + (t′−t)·v` at shift 0.5, t = (1, .6, .333, .143, 0). The 1-step DMD generator scores
  best on this grid (16.24, vs 16.28 for the student). Shipping a v2+ model needs `FEW_NFE_SAMPLER` changed
  accordingly in `model.py`.

Training uses backward simulation: a random step k, no-grad steps up to t_k, then the trained step.

| Run | Generator init | Grid | Differs from common | Raw FID (start → 125/250/375/500 kimg) | Best | Outcome |
|---|---|---|---|---|---|---|
| `dmd4_main` | student | v1 | lr 2e-6, TTUR 2, cfg 1.0 | 16.29 → –/20.56/–/19.84 (750: 20.96, 1000: 18.84) | 15.46 (EMA 500 @ 250, student/DMD blend) | Regressed; stopped at 1 Mimg |
| `dmd4b_euler_s05` | dmd1c best EMA 500 | v2 | as `dmd4_main` | 16.24 → –/18.05/–/19.98 | 15.99 | Regressed |
| `dmd4c_euler_s05_w15` | same | v2 | + guided real score (cfg 1.5) | 16.24 → –/22.57/–/24.26 | — | Worse: guidance is not the fix |
| `dmd4f_fakelr1e-5` | same | v2 | fake lr 1e-5 (generator 2e-6) | 16.24 → 16.55/17.02/17.34/17.54 | 15.78 | Slower regression |
| `dmd4h_fakelr3e-5` | same | v2 | fake lr 3e-5 | 16.25 → 15.96/16.19/15.82/16.01 | 15.56 | No regression; EMA improving |
| `dmd4i_fakelr1e-4` | same | v2 | fake lr 1e-4 | 16.24 → 15.51/**14.52**/… | 14.52 (raw @ 250) | Generator improves; running |

### What the Phase B runs taught us

- **Over-contrast, not lost diversity.**
  - *Symptom:* every regressing DMD generator overshoots into over-contrast. Per-image pixel std is 0.51–0.54
    against 0.476 for train images, with about +0.1 log10 excess power in every frequency band.
  - *Not diversity:* Inception precision rises and recall is flat (it drops only with the guided real score), so the FID increase
    comes from the mean term.
  - *Phase A too:* the Phase A "drift" shows the same signature (the 17.50 model has contrast 0.515).
  - *Diagnostics:* in the session scratchpad, not part of the submission.
- **The cause is fake-score lag.** The DM gradient `x_real − x_fake` keeps pointing toward more contrast. The fake
  score's picture of the generator lags behind, so it treats the generator's samples as softer than they are.
  - A faster fake score fixes it, monotonically: lr 2e-6 → 1e-5 → 3e-5 → 1e-4.
  - *Caveat:* the fake-score lr also drives the GAN head, so the discriminator is stronger at high fake lr. Part of
    the gain may come from that, and rising D logits are the drift risk to watch.

## Reproducibility gaps in the as-run lineage

1. **The 17.50 1-step model has a two-phase history.**
   - It was the P2 pilot for 639 kimg (cfg 1.5, 2.5 h budget), then continued with cfg 1.0 to 3,250 kimg and
     stopped by hand.
   - The result is the best evaluation (EMA 500 at 2,750 kimg), kept by `best.pt`.
   - `runs/dmd1c_p2cont/config.json` is the copied pilot config; the change is recorded in `continued.json` and in
     `scripts/dmd1c.sh`.
   - It has not been reproduced as a single command.
2. **Phase B v2+ inherits that history**, because its generator starts from `dmd1c_p2cont/best.pt`.
3. **Selection is by best-of evaluation** (`best.pt`, train-ref FID) plus manual stopping. The stopping points are
   recorded above.
4. **`dmd4i` ran as a pilot with `--snapshot_every_kimg 0`.** Its 14.52 weights at 250 kimg were not saved; only the
   final `latest.pt` remains.
5. **The DMD code evolved during the campaign.** Every change is an additive flag whose default reproduces the
   earlier behaviour:
   - `best.pt` saving, added after P2 had started;
   - `--update {rfpp,euler}` (Oct 14);
   - the contrast print, and no final snapshot when `--snapshot_every_kimg 0` (Oct 14).
6. **The DMD code and scripts were committed after the runs above** (2026-10-15). Those runs used the
   same code, apart from the additive flags in item 5.
7. **`rf2_main` was trained to 40 Mimg, then resumed to 60 Mimg** (`--total_kimg 60000`). The lr is constant and
   all state (optimizer, EMAs, RNG, data order) is restored, so this is equivalent to one 60 Mimg run.
8. **Intermediate files were deleted for disk space:** `fm_main` snapshots other than 30k, and `latest.pt` of
   finished or failed runs. This does not affect a from-scratch reproduction.

## Clean reproduction path

Stages 1–2 below are the commands that were actually run (the pipeline scripts add only scheduling and logging).
Stage 3 is the target: it should become one command per phase once the settings are final.

```bash
# Stage 1: teacher (~23.5 h). Defaults: lr 2e-4, batch 128, p_uncond 0.1, p_aug 0.15.
python train.py fm --run_dir runs/fm_main --arch unet79 --total_kimg 30000 --snapshot_every_kimg 3000 --fid_every_kimg 1500

# Stage 2a: guided teacher ODE pairs (~16.5 h)
python train.py pairs --snapshot runs/fm_main/snapshots/kimg_030000.pt --ema 2000 --out pairs/fm30k_euler64_w1.5 \
  --num_shards 20 --sampler euler --steps 64 --cfg 1.5 --interval 0.2,0.85
# Stage 2b: 2-RF++ student (~44 h); export kimg_060000 EMA 2000
python train.py reflow --run_dir runs/rf2_main --pairs pairs/fm30k_euler64_w1.5 \
  --init_snapshot runs/fm_main/snapshots/kimg_030000.pt --init_ema 2000 --total_kimg 60000

# Stage 3a (TARGET, not yet run in this form): 1-step DMD2 from the student, final settings in ONE run
python train.py dmd --run_dir runs/dmdA --ttur 2 --cfg 1.0 --lr_gen 2e-6 --lr_fake <final> \
  --time_budget_h <H> --fid_every_kimg 250 --snapshot_every_kimg 1500
# Stage 3b (TARGET): 4-step DMD2 from 3a's best generator
python train.py dmd --run_dir runs/dmdB --gen_init runs/dmdA/best.pt --gen_init_ema <key> --steps 4 --shift 0.5 \
  --update euler --ttur 2 --cfg 1.0 --lr_gen 2e-6 --lr_fake <final> --time_budget_h <H> --fid_every_kimg 250 \
  --snapshot_every_kimg 1500

# Export + checks
python export_ckpt.py --snapshot <best.pt> --ema <key> --modes one_nfe few_nfe --out_dir checkpoints/<name>
python check_submission.py
```

## Changelog

- 2026-10-15: file created. Running: `dmd4i_fakelr1e-4` (Phase B, fake lr 1e-4, to 500 kimg).
