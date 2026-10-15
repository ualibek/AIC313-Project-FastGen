#!/usr/bin/env bash
# DMD2 Phase A: four equal-time pilots (base + one change each), pick the main config, then the main run.
# Resume-aware: finished pilots (summary.json) are skipped; an interrupted run is resumed.
# Launch detached:  setsid nohup bash scripts/dmd_pipeline.sh > runs/dmd_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
PILOT_H=${PILOT_H:-2.5}
MAIN_H=${MAIN_H:-60}
MAIN=${MAIN:-runs/dmd1_main}
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

run() {  # run <run_dir> <flags...>: train (or resume) until it writes summary.json
  local dir=$1; shift
  if [ -f "${dir}/summary.json" ]; then echo "$(stamp) ${dir} already finished"; return 0; fi
  local resume=""
  [ -f "${dir}/latest.pt" ] && resume="--resume"
  echo "$(stamp) start ${dir} $* ${resume}"
  $PY train.py dmd --run_dir "${dir}" "$@" ${resume} >> "${dir}.log" 2>&1
  local code=$?
  echo "$(stamp) ${dir} exited with code ${code}"
  return ${code}
}

PILOT_ARGS="--time_budget_h ${PILOT_H} --fid_every_kimg 50 --snapshot_every_kimg 0 --latest_every_min 30"
P1_FLAGS="--lr_gen 1e-5 --lr_fake 1e-5"
P2_FLAGS="--ttur 2"
P3_FLAGS="--cfg 1.0"
run runs/dmd_p0_base ${PILOT_ARGS} || exit 1
run runs/dmd_p1_lr1e-5 ${PILOT_ARGS} ${P1_FLAGS}
run runs/dmd_p2_ttur2 ${PILOT_ARGS} ${P2_FLAGS}
run runs/dmd_p3_w1 ${PILOT_ARGS} ${P3_FLAGS}

MAIN_FLAGS=$($PY scripts/decide_dmd.py --base runs/dmd_p0_base --pilot runs/dmd_p1_lr1e-5 "${P1_FLAGS}" \
  --pilot runs/dmd_p2_ttur2 "${P2_FLAGS}" --pilot runs/dmd_p3_w1 "${P3_FLAGS}")
code=$?
echo "$(stamp) pilot decision (code ${code}): main flags '${MAIN_FLAGS}'"
[ "${code}" -eq 0 ] || exit 1

run "${MAIN}" --time_budget_h "${MAIN_H}" --fid_every_kimg 250 --snapshot_every_kimg 1500 ${MAIN_FLAGS} || exit 1
echo "$(stamp) Phase A main run done"
