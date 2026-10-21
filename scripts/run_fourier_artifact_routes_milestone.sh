#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: $0 OUTPUT_DIR [benchmark options]" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$1"; OUTPUT_DIR="$(cd "$1" && pwd)"; shift
if [[ -n "${PYTHON:-}" ]]; then PYTHON_BIN="$PYTHON"; elif command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
"$SCRIPT_DIR/run_covariant_milestone.sh" --output-dir "$OUTPUT_DIR" --model-metadata-json '{"benchmark":"fourier_artifact_routes"}' --python "$PYTHON_BIN" -- "$PYTHON_BIN" "$REPO_DIR/benchmark_fourier_artifact_routes.py" --output-dir "$OUTPUT_DIR" "$@"
"$PYTHON_BIN" "$REPO_DIR/plot_fourier_artifact_routes.py" --json "$OUTPUT_DIR/fourier_artifact_routes.json" --npz "$OUTPUT_DIR/fourier_artifact_routes.npz" --output-dir "$OUTPUT_DIR"
