#!/usr/bin/env bash
#
# run_server_all.sh — launch the full convergence-topology study on a server.
#
# Runs, in order:
#   1. Task 2: mass convergence-cluster analysis over every k-regular family,
#      every project size, and the supervisor scenarios.
#   2. Task 4: topological phase-transition sweeps (temperature 0..1) for a few
#      (k, n) combinations.
#
# Nothing here re-implements the learning core — it only drives the two Python
# entry points. All output lands under results/ (mirrors supervisor_results/).
#
# Usage:
#   bash experiments/exp8/gpu_version/run_server_all.sh
#
# Override any parameter via environment variables, e.g.:
#   ITERS=1000000 REPS=512 SIZES="10 20 50 100" \
#       bash experiments/exp8/gpu_version/run_server_all.sh
#
#   # only the phase-transition part:
#   STAGE=phase bash experiments/exp8/gpu_version/run_server_all.sh
#
#   # quick end-to-end check (tiny, seconds):
#   SMOKE=1 bash experiments/exp8/gpu_version/run_server_all.sh

set -euo pipefail

# ── locate repo root (this script lives in experiments/exp8/gpu_version) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PY="${PYTHON:-python3}"
PKG="experiments.exp8.gpu_version"

# ── tunable parameters (env-overridable) ──
STAGE="${STAGE:-all}"            # all | mass | phase
SMOKE="${SMOKE:-0}"             # 1 -> pass --smoke to both runners

ITERS="${ITERS:-500000}"
REPS="${REPS:-256}"
RECORD_EVERY="${RECORD_EVERY:-5000}"
N_FINAL_STEPS="${N_FINAL_STEPS:-10000}"
CLUSTER_METHOD="${CLUSTER_METHOD:-auto}"
SEED="${SEED:-42}"

# Task 2 — graph families / sizes / scenarios
GRAPHS="${GRAPHS:-ring cubic quartic quintic mixed23 mixed34}"
SIZES="${SIZES:-10 20 50}"
GAMMAS="${GAMMAS:-0.0 0.9}"
BETAS="${BETAS:-1.0}"
LEARNERS="${LEARNERS:-q_learning}"

# Task 4 — phase transitions: space-separated "k:n" pairs
PHASE_PAIRS="${PHASE_PAIRS:-2:20 3:20 4:20}"
PHASE_STEP="${PHASE_STEP:-0.05}"
PHASE_REALIZATIONS="${PHASE_REALIZATIONS:-3}"
PHASE_MODE="${PHASE_MODE:-stochastic}"
PHASE_GAMMA="${PHASE_GAMMA:-0.9}"
PHASE_BETA="${PHASE_BETA:-1.0}"
PHASE_LEARNER="${PHASE_LEARNER:-q_learning}"

LOG_DIR="${SCRIPT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

SMOKE_FLAG=""
if [[ "${SMOKE}" == "1" ]]; then
  SMOKE_FLAG="--smoke"
fi

echo "=================================================================="
echo "  Convergence-topology server run"
echo "  repo_root : ${REPO_ROOT}"
echo "  python    : ${PY}"
echo "  stage     : ${STAGE}   smoke=${SMOKE}"
echo "  logs      : ${LOG_DIR}"
echo "=================================================================="

run_mass() {
  echo ">>> [Task 2] mass convergence-topology experiments"
  ${PY} -m "${PKG}.run_all_convergence_topology_experiments" \
    ${SMOKE_FLAG} \
    --graphs ${GRAPHS} \
    --sizes ${SIZES} \
    --gammas ${GAMMAS} \
    --betas ${BETAS} \
    --learners ${LEARNERS} \
    --iters "${ITERS}" \
    --reps "${REPS}" \
    --record-every "${RECORD_EVERY}" \
    --n-final-steps "${N_FINAL_STEPS}" \
    --cluster-method "${CLUSTER_METHOD}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/mass_${STAMP}.log"
}

run_phase() {
  echo ">>> [Task 4] topology phase-transition sweeps: ${PHASE_PAIRS}"
  for pair in ${PHASE_PAIRS}; do
    k="${pair%%:*}"
    n="${pair##*:}"
    echo "    --- phase transition k=${k} -> $((k + 1)), n=${n} ---"
    ${PY} -m "${PKG}.run_topology_phase_transition" \
      ${SMOKE_FLAG} \
      --n "${n}" --k "${k}" \
      --step "${PHASE_STEP}" \
      --realizations "${PHASE_REALIZATIONS}" \
      --mode "${PHASE_MODE}" \
      --gamma "${PHASE_GAMMA}" \
      --beta "${PHASE_BETA}" \
      --learner "${PHASE_LEARNER}" \
      --iters "${ITERS}" \
      --reps "${REPS}" \
      --record-every "${RECORD_EVERY}" \
      --n-final-steps "${N_FINAL_STEPS}" \
      --cluster-method "${CLUSTER_METHOD}" \
      --seed "${SEED}" \
      2>&1 | tee "${LOG_DIR}/phase_k${k}_n${n}_${STAMP}.log"
  done
}

case "${STAGE}" in
  mass)  run_mass ;;
  phase) run_phase ;;
  all)   run_mass; run_phase ;;
  *) echo "Unknown STAGE='${STAGE}' (expected all|mass|phase)"; exit 1 ;;
esac

echo "=================================================================="
echo "  Done. Results under: ${SCRIPT_DIR}/results/"
echo "=================================================================="
