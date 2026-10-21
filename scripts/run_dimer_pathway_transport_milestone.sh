#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR RECOVERY_JSON [benchmark options]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "$1"
OUTPUT_DIR="$(cd "$1" && pwd)"
RECOVERY_JSON="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
shift 2
if [[ -n "${PYTHON:-}" ]]; then PYTHON_BIN="$PYTHON";
elif command -v python >/dev/null 2>&1; then PYTHON_BIN=python;
else PYTHON_BIN=python3; fi

exec "${SCRIPT_DIR}/run_covariant_milestone.sh" \
  --output-dir "${OUTPUT_DIR}" \
  --model-metadata-json '{"benchmark":"ordered_liouville_pathway_transport","controls":["J_12","k_12","gamma_phi"]}' \
  --python "${PYTHON_BIN}" -- \
  "${PYTHON_BIN}" "${REPO_DIR}/benchmark_dimer_pathway_parallel_transport.py" \
  --recovery-json "${RECOVERY_JSON}" --output-dir "${OUTPUT_DIR}" "$@"
