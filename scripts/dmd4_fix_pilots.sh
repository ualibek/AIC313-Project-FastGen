#!/usr/bin/env bash
# DMD2 Phase B over-contrast fix: 500-kimg A/B pilots against runs/dmd4b_euler_s05 (raw 4-step train-ref FID
# 16.25 -> 18.06 @250 -> 19.98 @500 kimg). Diagnosis: every DMD generator overshoots to over-contrast (sample std
# 0.51-0.54 vs 0.476 for train images, +0.09 log10 power in all bands), and the DM update x_real - x_fake still points
# toward more contrast on over-contrasted samples (cos +0.1..+0.2 at t 0.3-0.85). Reading: the fake score lags the
# generator, so it sees G's samples as softer than they are and x_real - x_fake keeps adding contrast. => make the
# fake score track faster:
#   fakelr: fake-score lr 2e-6 -> 1e-5 (G stays 2e-6), TTUR 2
#   ttur5 : 5 fake updates per G update (lr 2e-6), 375 kimg -- dropped before it ran (scripts/dmd4_fix_pilots2.sh)
# Pilots run with --snapshot_every_kimg 0 (no best.pt / final snapshot; latest.pt keeps G + EMAs).
# Everything else equals dmd4b (G from dmd1c best.pt EMA 500, 4-step Euler grid shift 0.5, fresh fake, lr 2e-6,
# TTUR 2, cfg 1.0). Resume-aware. Launch detached:
#   setsid nohup bash scripts/dmd4_fix_pilots.sh > runs/dmd4_fix_pilots.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
COMMON="--gen_init runs/dmd1c_p2cont/best.pt --gen_init_ema 500 --steps 4 --shift 0.5 --update euler
  --lr_gen 2e-6 --lr_fake 2e-6 --ttur 2 --cfg 1.0 --total_kimg 500 --fid_every_kimg 125 --snapshot_every_kimg 1500"
run() {  # run_dir, extra args
  local dir=$1; shift
  [ -f "${dir}/summary.json" ] && { echo "${dir} already finished"; return; }
  local resume=""
  [ -f "${dir}/latest.pt" ] && resume="--resume"
  echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") start ${dir} $* ${resume}"
  $PY train.py dmd --run_dir "${dir}" ${COMMON} "$@" ${resume} >> "${dir}.log" 2>&1
  local code=$?
  echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${dir} exited with code ${code}"
}
run runs/dmd4f_fakelr1e-5 --lr_fake 1e-5 --snapshot_every_kimg 0
