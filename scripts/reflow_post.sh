#!/usr/bin/env bash
# After the main 2-RF++ run: few-step sampler sweep, then config (g) of Lee et al. (2024) -- invert the real
# train images with the teacher ODE and fine-tune the student on (inverted noise, real image) pairs only.
# Launch detached:  WAIT_PID=<orchestrator pid> setsid nohup bash scripts/reflow_post.sh > runs/reflow_post.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
MAIN=${MAIN:-runs/rf2_main}
SNAP=${SNAP:-${MAIN}/snapshots/kimg_040000.pt}
TEACHER=${TEACHER:-runs/fm_main/snapshots/kimg_030000.pt}
TEACHER_EMA=${TEACHER_EMA:-2000}
TEACHER_SAMPLER=${TEACHER_SAMPLER:---sampler euler --steps 64 --cfg 1.5 --interval 0.2,0.85}
REAL_PAIRS=${REAL_PAIRS:-pairs/fm30k_real_inv}
REAL_RUN=${REAL_RUN:-runs/rf2_real}
REAL_KIMG=${REAL_KIMG:-1280}  # 5k steps x 256, as in the paper
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

# Best EMA at the snapshot's final evaluation (1-NFE train-reference FID).
best_ema() {
  awk -F, -v k="$2" '$2==k && $5=="euler1" {print $7, $4}' "$1/fid.csv" | sort -g | head -1 | awk '{printf "%g", $2}'
}

if [ -n "${WAIT_PID:-}" ]; then
  echo "$(stamp) waiting for PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
fi
[ -f "${SNAP}" ] || { echo "$(stamp) ${SNAP} missing, stopping"; exit 1; }
SNAP_KIMG=$(basename "${SNAP}" .pt | sed 's/kimg_0*//')
EMA=$(best_ema "${MAIN}" "${SNAP_KIMG}")
[ -n "${EMA}" ] || { echo "$(stamp) no FID rows for ${SNAP_KIMG} kimg in ${MAIN}/fid.csv"; exit 1; }
echo "$(stamp) ${SNAP}: best EMA ${EMA}"

echo "$(stamp) student sweep"
$PY eval_fm.py --snapshot "${SNAP}" --ema "${EMA}" --preset student >> "${MAIN}/eval_student.log" 2>&1
echo "$(stamp) student sweep exited with code $?"

if [ ! -f "${REAL_PAIRS}/x_0000.npy" ]; then
  echo "$(stamp) inverting real images -> ${REAL_PAIRS}"
  $PY train.py pairs --invert --snapshot "${TEACHER}" --ema "${TEACHER_EMA}" --out "${REAL_PAIRS}" ${TEACHER_SAMPLER} \
    >> "${REAL_PAIRS}.log" 2>&1
  code=$?
  echo "$(stamp) inversion exited with code ${code}"
  [ "${code}" -eq 0 ] || exit 1
fi

RESUME=""
[ -f "${REAL_RUN}/latest.pt" ] && RESUME="--resume"
echo "$(stamp) real-pair fine-tune ${REAL_RUN} from ${SNAP} [ema ${EMA}]"
$PY train.py reflow --run_dir "${REAL_RUN}" --pairs "${REAL_PAIRS}" --init_snapshot "${SNAP}" --init_ema "${EMA}" \
  --lr 1e-5 --warmup_steps 0 --total_kimg "${REAL_KIMG}" --ema_halflives_kimg 50,250 \
  --fid_every_kimg 256 --snapshot_every_kimg 640 ${RESUME} >> "${REAL_RUN}.log" 2>&1
code=$?
echo "$(stamp) fine-tune exited with code ${code}"
[ "${code}" -eq 0 ] || exit 1

REAL_SNAP=$(printf "%s/snapshots/kimg_%06d.pt" "${REAL_RUN}" "${REAL_KIMG}")
REAL_EMA=$(best_ema "${REAL_RUN}" "${REAL_KIMG}")
echo "$(stamp) student sweep on ${REAL_SNAP} [ema ${REAL_EMA}]"
$PY eval_fm.py --snapshot "${REAL_SNAP}" --ema "${REAL_EMA}" --preset student >> "${REAL_RUN}/eval_student.log" 2>&1
echo "$(stamp) done (code $?)"
