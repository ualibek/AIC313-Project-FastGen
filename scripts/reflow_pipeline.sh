#!/usr/bin/env bash
# Reflow pipeline: teacher ODE pairs -> 2-rectified flow++ training, back to back (never sharing the GPU).
# Resume-aware: finished pair shards are skipped and training resumes from latest.pt.
# Launch detached:  setsid nohup bash scripts/reflow_pipeline.sh > runs/reflow_orchestrator.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
TEACHER=${TEACHER:-runs/fm_main/snapshots/kimg_030000.pt}
TEACHER_EMA=${TEACHER_EMA:-2000}
PAIRS=${PAIRS:?set PAIRS=pairs/<name>}
PAIR_ARGS=${PAIR_ARGS:?set PAIR_ARGS to the teacher sampler flags}
NUM_SHARDS=${NUM_SHARDS:-20}
RUN=${RUN:-runs/rf2_main}
RUN_ARGS=${RUN_ARGS:-}
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
mkdir -p "$(dirname "${PAIRS}")" runs

echo "$(stamp) pairs: ${PAIRS} (${PAIR_ARGS}, ${NUM_SHARDS} shards)"
$PY train.py pairs --snapshot "${TEACHER}" --ema "${TEACHER_EMA}" --out "${PAIRS}" --num_shards "${NUM_SHARDS}" \
  ${PAIR_ARGS} >> "${PAIRS}.log" 2>&1
code=$?
done_shards=$(ls "${PAIRS}"/x_[0-9][0-9][0-9][0-9].npy 2>/dev/null | wc -l)
echo "$(stamp) pair generation exited with code ${code}; ${done_shards}/${NUM_SHARDS} shards"
if [ "${code}" -ne 0 ] || [ "${done_shards}" -lt "${NUM_SHARDS}" ]; then
  echo "$(stamp) not starting training"
  exit 1
fi

RESUME=""
[ -f "${RUN}/latest.pt" ] && RESUME="--resume"
echo "$(stamp) starting reflow training ${RUN}"
$PY train.py reflow --run_dir "${RUN}" --pairs "${PAIRS}" --init_snapshot "${TEACHER}" --init_ema "${TEACHER_EMA}" \
  ${RUN_ARGS} ${RESUME} >> "${RUN}.log" 2>&1
echo "$(stamp) reflow training exited with code $?"
