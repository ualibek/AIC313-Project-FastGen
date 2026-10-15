#!/usr/bin/env bash
# DMD2 Phase B, second design: 4-step generator for ModelFewNFE, started from the Phase A 1-step DMD generator.
# Why: runs/dmd4_main (student init, RF++ update, shift 2) first got worse than the student (16.3 -> ~19-21) and
# recovered slowly. The 1-step DMD generator already beats the student when sampled with plain Euler steps at low
# shift (train-ref FID: 3-step shift 0.5 15.85, 4-step shift 0.5 16.24 vs student 16.28): its first step does the
# work and the later low-noise steps refine. This trains exactly that 4-step grid, (1, 0.6, 0.333, 0.143, 0) with
# the Euler update, by backward simulation; step-0 FID must reproduce 16.24.
# Fresh fake score + discriminator from the teacher (500 fake-only warm-up), Phase A's stable settings.
# Result (cfg 1.0, runs/dmd4b_euler_s05, stopped at 500 kimg): raw 16.25 -> 18.06 -> 19.98, the same regression as
# dmd4_main. Reading: DMD pulls G toward the real score's distribution, and the unguided teacher is only ~16.6
# (Euler-32 train-ref), i.e. no better than G already is. The guided teacher (w=1.5 on [0.2, 0.85]) reaches 13.5,
# so CFG=1.5 (the trainer's default guidance and interval) gives DMD a target that is better than G.
# Resume-aware. Launch detached:
#   DIR=runs/dmd4c_euler_s05_w15 CFG=1.5 setsid nohup bash scripts/dmd4b.sh > runs/dmd4c_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
DIR=${DIR:-runs/dmd4b_euler_s05}
HOURS=${HOURS:-36}
CFG=${CFG:-1.0}
[ -f "${DIR}/summary.json" ] && { echo "${DIR} already finished"; exit 0; }
resume=""
[ -f "${DIR}/latest.pt" ] && resume="--resume"
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") start ${DIR} cfg ${CFG} ${resume}"
$PY train.py dmd --run_dir "${DIR}" --gen_init runs/dmd1c_p2cont/best.pt --gen_init_ema 500 \
  --steps 4 --shift 0.5 --update euler --lr_gen 2e-6 --lr_fake 2e-6 --ttur 2 --cfg "${CFG}" \
  --time_budget_h "${HOURS}" --fid_every_kimg 250 --snapshot_every_kimg 1500 ${resume} >> "${DIR}.log" 2>&1
code=$?
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${DIR} exited with code ${code}"
