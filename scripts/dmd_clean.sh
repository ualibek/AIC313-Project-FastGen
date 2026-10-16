#!/usr/bin/env bash
# Stage 3 of the clean reproduction path (EXPERIMENTS.md): DMD2 from the 2-RF++ student in two single-command runs,
# with the settings the pilots selected (generator lr 2e-6, TTUR 2, unguided real score):
#   Run A: 1-step generator (ModelOneNFE), init = student rf2_main 60k EMA 2000, fake-score lr 2e-6, 3,000 kimg.
#          (fake lr 1e-4 from the student stalled at ~19 FID by 1 Mimg with a much stronger discriminator, since the
#          GAN head shares the fake-score lr: runs/dmdA_fakelr1e-4. The Phase A setting that reached 17.50 is used.)
#   Run B: 4-step generator (ModelFewNFE), init = Run A's best.pt (the evaluated best of raw / EMAs), 4 Euler steps at
#          shift 0.5 with backward simulation, fake-score lr 1e-4 (stops the over-contrast drift), 2,000 kimg
# Both: real score = teacher fm_main 30k EMA 2000 (frozen), fresh fake score + GAN head from the teacher, fixed
# length, selection = best.pt (best train-reference FID over all evaluations of raw and EMA weights).
# Each run directory gets the git commit and the uncommitted diff it was launched with. Resume-aware.
# Launch detached:  setsid nohup bash scripts/dmd_clean.sh > runs/dmd_clean.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
A=${A:-runs/dmdA}
B=${B:-runs/dmdB}
COMMON="--lr_gen 2e-6 --ttur 2 --cfg 1.0 --fid_every_kimg 250 --snapshot_every_kimg 1500"
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
run() {  # run <run_dir> <flags...>: train (or resume) until it writes summary.json
  local dir=$1; shift
  if [ -f "${dir}/summary.json" ]; then echo "$(stamp) ${dir} already finished"; return 0; fi
  local resume=""
  [ -f "${dir}/latest.pt" ] && resume="--resume"
  mkdir -p "${dir}"
  if [ -z "${resume}" ]; then
    git rev-parse HEAD > "${dir}/git_commit.txt"
    { git diff HEAD; git ls-files --others --exclude-standard | while read -r f; do git diff --no-index /dev/null "$f"; done; } \
      > "${dir}/git_diff.patch"
  fi
  echo "$(stamp) start ${dir} $* ${resume}"
  $PY train.py dmd --run_dir "${dir}" "$@" ${resume} >> "${dir}.log" 2>&1
  local code=$?
  echo "$(stamp) ${dir} exited with code ${code}"
  return ${code}
}
run "${A}" ${COMMON} --lr_fake 2e-6 --total_kimg 3000 || exit 1
run "${B}" ${COMMON} --lr_fake 1e-4 --gen_init "${A}/best.pt" --gen_init_ema best --steps 4 --shift 0.5 --update euler \
  --total_kimg 2000 || exit 1
echo "$(stamp) clean DMD path done"
