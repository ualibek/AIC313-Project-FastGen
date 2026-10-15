#!/usr/bin/env bash
# DMD2 Phase B: 4-step generator for ModelFewNFE (backward simulation), with Phase A's stable settings
# (lr 2e-6, TTUR 2, unguided real score). The generator starts from the 2-RF student, which is trained at every t;
# the fake score starts from the teacher with the usual 500 fake-only warm-up (a fresh discriminator: the drift seen
# in Phase A grows with the discriminator's age). Time grid = FEW_NFE_SAMPLER (4 steps, shift 2, RF++ update);
# the step-0 FID must reproduce the student's 4-NFE train-reference FID (16.28).
# Resume-aware. Launch detached:  setsid nohup bash scripts/dmd4.sh > runs/dmd4_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
DIR=${DIR:-runs/dmd4_main}
HOURS=${HOURS:-48}
[ -f "${DIR}/summary.json" ] && { echo "${DIR} already finished"; exit 0; }
resume=""
[ -f "${DIR}/latest.pt" ] && resume="--resume"
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") start ${DIR} ${resume}"
$PY train.py dmd --run_dir "${DIR}" --steps 4 --shift 2 --lr_gen 2e-6 --lr_fake 2e-6 --ttur 2 --cfg 1.0 \
  --time_budget_h "${HOURS}" --fid_every_kimg 250 --snapshot_every_kimg 1500 ${resume} >> "${DIR}.log" 2>&1
code=$?
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${DIR} exited with code ${code}"
