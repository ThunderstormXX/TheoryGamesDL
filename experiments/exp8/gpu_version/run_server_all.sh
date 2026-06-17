#!/usr/bin/env bash
#
# run_server_all.sh — launch the full convergence-topology study on a server.
#
# Tuned for a 40 GB A100: large VRAM-filling batches (--auto-reps), the fast
# shared-adjacency matmul reward (default in the simulation), and a process pool
# (--workers) so the many independent runs proceed in parallel.
#
# Runs, in order:
#   1. Task 2: mass convergence-cluster analysis over every k-regular family
#      (incl. the larger mixed graphs), every size, and a rich gamma/beta grid.
#   2. Task 4: topological phase-transition sweeps (temperature 0..1).
#
# Nothing here re-implements the learning core — it only drives the two Python
# entry points. All output lands under results/ (mirrors supervisor_results/).
#
# Usage:
#   bash experiments/exp8/gpu_version/run_server_all.sh                 # A100 defaults
#
# Override any parameter via environment variables, e.g.:
#   ITERS=1000000 WORKERS=4 bash experiments/exp8/gpu_version/run_server_all.sh
#   AUTO_REPS=0 REPS=8192 bash experiments/exp8/gpu_version/run_server_all.sh
#   STAGE=phase bash experiments/exp8/gpu_version/run_server_all.sh
#   SMOKE=1     bash experiments/exp8/gpu_version/run_server_all.sh      # tiny check

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
RECORD_EVERY="${RECORD_EVERY:-5000}"
N_FINAL_STEPS="${N_FINAL_STEPS:-10000}"
CLUSTER_METHOD="${CLUSTER_METHOD:-auto}"
SEED="${SEED:-42}"

# A100 batch / parallelism controls
AUTO_REPS="${AUTO_REPS:-1}"          # 1 -> size reps to fill VRAM
VRAM_FRACTION="${VRAM_FRACTION:-0.85}"
MAX_REPS="${MAX_REPS:-131072}"
REPS="${REPS:-8192}"                  # used only when AUTO_REPS=0
WORKERS="${WORKERS:-1}"              # parallel processes sharing the GPU
STORE_REPS="${STORE_REPS:-reduced}"  # reduced | full

# Task 2 — graph families / sizes / scenarios (rich grid by default)
GRAPHS="${GRAPHS:-ring cubic quartic quintic mixed23 mixed34 mixed45 mixed56}"
SIZES="${SIZES:-10 20 50 100}"
GAMMAS="${GAMMAS:-0.0 0.5 0.8 0.9 0.95 0.99}"
BETAS="${BETAS:-0.5 1.0 2.0 4.0}"
LEARNERS="${LEARNERS:-q_learning}"

# Task 4 — phase transitions: space-separated "k:n" pairs
PHASE_PAIRS="${PHASE_PAIRS:-2:20 3:20 4:20 5:20}"
PHASE_STEP="${PHASE_STEP:-0.05}"
PHASE_REALIZATIONS="${PHASE_REALIZATIONS:-5}"
PHASE_MODE="${PHASE_MODE:-stochastic}"
PHASE_GAMMA="${PHASE_GAMMA:-0.9}"
PHASE_BETA="${PHASE_BETA:-1.0}"
PHASE_LEARNER="${PHASE_LEARNER:-q_learning}"

LOG_DIR="${SCRIPT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

SMOKE_FLAG=""
[[ "${SMOKE}" == "1" ]] && SMOKE_FLAG="--smoke"

REPS_FLAG=("--reps" "${REPS}")
[[ "${AUTO_REPS}" == "1" ]] && REPS_FLAG=("--auto-reps" "--vram-fraction" "${VRAM_FRACTION}" "--max-reps" "${MAX_REPS}")

echo "=================================================================="
echo "  Convergence-topology server run (A100-tuned)"
echo "  repo_root : ${REPO_ROOT}"
echo "  stage=${STAGE} smoke=${SMOKE} workers=${WORKERS} auto_reps=${AUTO_REPS}"
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
    "${REPS_FLAG[@]}" \
    --workers "${WORKERS}" \
    --store-reps "${STORE_REPS}" \
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
      "${REPS_FLAG[@]}" \
      --workers "${WORKERS}" \
      --store-reps "${STORE_REPS}" \
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
