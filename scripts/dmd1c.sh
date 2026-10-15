#!/usr/bin/env bash
# DMD2 Phase A, third attempt: continue pilot P2 (lr 2e-6, TTUR 2; healthy discriminator, still improving at 639 kimg)
# with the unguided real score (cfg 1.0, pilot P3) as a long run.
# Why: runs/dmd1_main (lr 1e-5) peaked at 750 kimg and then drifted (over-contrast; low-noise discriminator logits
# grew to +-4); restarting it at lr 2e-6 (runs/dmd1b_lr2e-6) inherited that over-confident discriminator and did not
# improve. P2's own state never drifted.
# The pilot's directory is copied (not modified) on the first launch; afterwards this script just resumes.
# Launch detached:  setsid nohup bash scripts/dmd1c.sh > runs/dmd1c_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/pokemon-generation/bin/python
SRC=runs/dmd_p2_ttur2
DIR=${DIR:-runs/dmd1c_p2cont}
HOURS=${HOURS:-42}  # total training hours, including P2's 2.5
[ -f "${DIR}/summary.json" ] && { echo "${DIR} already finished"; exit 0; }
if [ ! -f "${DIR}/latest.pt" ]; then
  mkdir -p "${DIR}/samples" "${DIR}/snapshots"
  cp "${SRC}/latest.pt" "${SRC}/log.csv" "${SRC}/fid.csv" "${SRC}/config.json" "${DIR}/"
  echo "{\"continued_from\": \"${SRC}\", \"changed\": \"--cfg 1.0, --fid_every_kimg 250, --snapshot_every_kimg 1500, --time_budget_h ${HOURS}\"}" > "${DIR}/continued.json"
fi
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") resume ${DIR}"
$PY train.py dmd --run_dir "${DIR}" --ttur 2 --cfg 1.0 --time_budget_h "${HOURS}" --fid_every_kimg 250 \
  --snapshot_every_kimg 1500 --resume >> "${DIR}.log" 2>&1
code=$?
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${DIR} exited with code ${code}"
