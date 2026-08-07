#!/usr/bin/env bash
set -euo pipefail

python prepare_sake_dimer_derivatives.py \
  --omega0 2.0 \
  --delta-omega 0.18 \
  --reference-J 0.0 \
  --reference-k 0.0 \
  --max-order 2 \
  --output dimer_backend_benchmarks/sake_dimer_zero_cache.npz

common=(
  --derivative-cache dimer_backend_benchmarks/sake_dimer_zero_cache.npz
  --omega0 2.0
  --delta-omega 0.18
  --J-min -0.15
  --J-max 0.15
  --k-min 0.0
  --k-max 0.03
  --n-omega 8
  --n-quad 4
  --max-iterations 30
  --initial-radius 0.005
  --maximum-radius 0.010
  --minimum-radius 0.00025
  --parameter-tolerance 0.00025
)

python fit_sake_dimer_iterative.py "${common[@]}" \
  --target-J -0.100 --target-k 0.012 \
  --output-dir dimer_backend_benchmarks/iterative_inverse_target_bounded_n8

python fit_sake_dimer_iterative.py "${common[@]}" \
  --target-J 0.080 --target-k 0.008 \
  --output-dir dimer_backend_benchmarks/iterative_inverse_positive_J_n8

python fit_sake_dimer_iterative.py "${common[@]}" \
  --target-J -0.060 --target-k 0.020 \
  --output-dir dimer_backend_benchmarks/iterative_inverse_mixed_n8

python plot_sake_multi_target_recovery.py \
  dimer_backend_benchmarks/iterative_inverse_target_bounded_n8 \
  dimer_backend_benchmarks/iterative_inverse_positive_J_n8 \
  dimer_backend_benchmarks/iterative_inverse_mixed_n8 \
  --omega0 2.0 \
  --normalize-controls \
  --output-dir dimer_backend_benchmarks/iterative_inverse_multi_target_n8
