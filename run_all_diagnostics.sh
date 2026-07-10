#!/usr/bin/env bash
set -euo pipefail

python diagnose_reference_basis.py
python diagnose_transport_stages.py
