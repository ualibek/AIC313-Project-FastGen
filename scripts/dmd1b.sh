#!/usr/bin/env bash
# DMD2 Phase A restart. runs/dmd1_main (pilot-chosen lr 1e-5 + TTUR 2 + cfg 1.0) peaked at 750 kimg (EMA 200,
# FID 18.03) and then drifted (raw 18.6 -> 22.1 by 1250 kimg, D logits growing, as in pilot P1), so it was stopped.
# This continues from its best evaluation at the stable pilot lr 2e-6, taking the fake score + GAN head from its
# latest.pt (500 fake-only warm-up iterations re-fit them to the restored generator).
# Resume-aware. Launch detached:  setsid nohup bash scripts/dmd1b.sh > runs/dmd1b_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
DIR=${DIR:-runs/dmd1b_lr2e-6}
HOURS=${HOURS:-48}
[ -f "${DIR}/summary.json" ] && { echo "${DIR} already finished"; exit 0; }
resume=""
[ -f "${DIR}/latest.pt" ] && resume="--resume"
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") start ${DIR} ${resume}"
$PY train.py dmd --run_dir "${DIR}" --gen_init runs/dmd1_main/best.pt --gen_init_ema 200 \
  --fake_from runs/dmd1_main/latest.pt --lr_gen 2e-6 --lr_fake 2e-6 --ttur 2 --cfg 1.0 \
  --time_budget_h "${HOURS}" --fid_every_kimg 250 --snapshot_every_kimg 1500 ${resume} >> "${DIR}.log" 2>&1
code=$?
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${DIR} exited with code ${code}"
