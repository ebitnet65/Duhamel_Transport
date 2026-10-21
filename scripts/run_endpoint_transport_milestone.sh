#!/usr/bin/env bash
# Portable Mac/Linux wrapper for the fixed-endpoint route benchmark.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_endpoint_transport_milestone.sh OUTPUT_DIR \
    (--endpoint J K GAMMA | --endpoint-json RECOVERY.json) [benchmark options]

The command runs from any working directory. It writes runtime_metadata.json,
run.log, transport_diagnostics.json, and transport_paths.npz to OUTPUT_DIR.
EOF
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="$1"
shift
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

exec "${SCRIPT_DIR}/run_covariant_milestone.sh" \
  --output-dir "${OUTPUT_DIR}" \
  --model-metadata-json '{"benchmark":"fixed_endpoint_dimer_route","controls":["J_12","k_12","gamma_phi"]}' \
  --python "${PYTHON_BIN}" \
  -- "${PYTHON_BIN}" "${REPO_DIR}/benchmark_recovered_endpoint_transport.py" \
  --output-dir "${OUTPUT_DIR}" "$@"
