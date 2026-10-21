#!/usr/bin/env bash
# Run a covariant milestone command with portable provenance and captured logs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_covariant_milestone.sh --output-dir DIR [options] -- COMMAND [ARGS...]

Options:
  --model-metadata-json JSON_OR_FILE  Model/run metadata object or JSON file.
  --seed NAME=VALUE                  Record a seed (repeatable).
  --python PYTHON                    Python interpreter (default: $PYTHON, python, or python3).
  -h, --help                         Show this help.

The launcher is independent of the current directory, forces headless plotting,
enables JAX 64-bit mode by default, records runtime_metadata.json, and captures
combined output in run.log.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR=""
MODEL_METADATA=""
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
SEEDS=()

while (($#)); do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"; shift 2 ;;
    --model-metadata-json)
      [[ $# -ge 2 ]] || { echo "--model-metadata-json requires a value" >&2; exit 2; }
      MODEL_METADATA="$2"; shift 2 ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed requires NAME=VALUE" >&2; exit 2; }
      SEEDS+=("$2"); shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      PYTHON_BIN="$2"; shift 2 ;;
    --)
      shift; break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "COMMAND is required after --" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2; exit 127;
}

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
METADATA_PATH="${OUTPUT_DIR}/runtime_metadata.json"
LOG_PATH="${OUTPUT_DIR}/run.log"
STARTED_AT="$(${PYTHON_BIN} -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"

metadata_args=(
  --output "${METADATA_PATH}"
  --repo-dir "${REPO_DIR}"
  --started-at "${STARTED_AT}"
)
if [[ -n "${MODEL_METADATA}" ]]; then
  metadata_args+=(--model-metadata-json "${MODEL_METADATA}")
fi
if ((${#SEEDS[@]})); then
  for seed in "${SEEDS[@]}"; do
    metadata_args+=(--seed "${seed}")
  done
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"

"${PYTHON_BIN}" "${REPO_DIR}/runtime_metadata.py" "${metadata_args[@]}" -- "$@" >/dev/null

set +e
(cd "${REPO_DIR}" && "$@") 2>&1 | tee "${LOG_PATH}"
run_status=${PIPESTATUS[0]}
set -e

FINISHED_AT="$(${PYTHON_BIN} -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"
"${PYTHON_BIN}" "${REPO_DIR}/runtime_metadata.py" \
  "${metadata_args[@]}" \
  --finished-at "${FINISHED_AT}" \
  --exit-code "${run_status}" \
  -- "$@" >/dev/null

exit "${run_status}"
