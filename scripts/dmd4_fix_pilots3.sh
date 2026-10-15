#!/usr/bin/env bash
# Second batch of Phase B over-contrast pilots (see scripts/dmd4_fix_pilots.sh for the diagnosis).
# runs/dmd4f_fakelr1e-5 (fake lr 1e-5, G 2e-6, TTUR 2) confirmed the lag reading: raw 16.24 -> 16.56 @125 ->
# 17.02 @250 kimg (vs 18.06 @250 at fake lr 2e-6), EMA200/500 15.80 @250, contrast
# levelling at ~0.52 (train images 0.476). TTUR 5 at lr 2e-6 tracks slower per G step (5 x 2e-6 < 2 x 1e-5) at 2.5x
# the cost, so it is dropped for a faster fake score:
#   fakelr3e-5: fake lr 3e-5 (G 2e-6, TTUR 2). RESULT @250: raw 15.96 @125 (first raw gain over the init), 16.19 @250,
#               EMA200 15.68; contrast 0.511-0.520. Faster is better so far; the teacher itself trained at lr 2e-4:
#   fakelr1e-4: fake lr 1e-4
# Waits for any running DMD training (the dmd4h pilot) to finish first. Resume-aware. Launch detached:
#   setsid nohup bash scripts/dmd4_fix_pilots3.sh > runs/dmd4_fix_pilots3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
while pgrep -f "^${PY} train.py dmd" > /dev/null; do sleep 60; done
COMMON="--gen_init runs/dmd1c_p2cont/best.pt --gen_init_ema 500 --steps 4 --shift 0.5 --update euler
  --lr_gen 2e-6 --ttur 2 --cfg 1.0 --total_kimg 500 --fid_every_kimg 125 --snapshot_every_kimg 0"
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
run runs/dmd4h_fakelr3e-5 --lr_fake 3e-5
run runs/dmd4i_fakelr1e-4 --lr_fake 1e-4
