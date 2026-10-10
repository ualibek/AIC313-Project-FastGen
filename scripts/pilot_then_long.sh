#!/usr/bin/env bash
# Overnight pipeline: two equal-time architecture pilots -> pick arch -> long flow-matching run.
# Launch detached:  setsid nohup bash scripts/pilot_then_long.sh > runs/orchestrator.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
mkdir -p runs
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

PILOT_ARGS="--time_budget_h 2.0 --total_kimg 100000 --fid_every_kimg 500 --snapshot_every_kimg 0 --ema_halflives_kimg 500"
for ARCH in unet68 unet79; do
  if [ -f "runs/pilot_${ARCH}/summary.json" ]; then
    echo "$(stamp) pilot ${ARCH} already done"
    continue
  fi
  RESUME=""
  [ -f "runs/pilot_${ARCH}/latest.pt" ] && RESUME="--resume"
  echo "$(stamp) starting pilot ${ARCH}"
  $PY train.py fm --run_dir "runs/pilot_${ARCH}" --arch "${ARCH}" ${PILOT_ARGS} ${RESUME} \
    >> "runs/pilot_${ARCH}.log" 2>&1
  echo "$(stamp) pilot ${ARCH} exited with code $?"
done

CHOICE=$($PY scripts/decide_arch.py | tail -n 1)
echo "$(stamp) decision: ${CHOICE}"
if [ "${CHOICE}" = "none" ]; then
  echo "$(stamp) both pilots failed; not launching the long run"
  exit 1
fi
if [ "${CHOICE}" = "unet68" ]; then
  LONG_ARGS="--total_kimg 26000 --snapshot_every_kimg 2600 --fid_every_kimg 1300"  # ~same wall-clock as unet79
else
  LONG_ARGS="--total_kimg 30000 --snapshot_every_kimg 3000 --fid_every_kimg 1500"
fi
RESUME=""
[ -f runs/fm_main/latest.pt ] && RESUME="--resume"
echo "$(stamp) starting long run fm_main (${CHOICE})"
$PY train.py fm --run_dir runs/fm_main --arch "${CHOICE}" ${LONG_ARGS} ${RESUME} >> runs/fm_main.log 2>&1
echo "$(stamp) long run exited with code $?"
