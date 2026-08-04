# SAKE Autodiff Dimer Release Branch

This branch is the development and publication branch for SAKE:

```text
SAKE = Spectral-Autodiff Kernel Evaluation
```

It contains the four-level `su(2) x su(2)` exciton-dimer model, the JAX
autodiff Liouvillian derivative backend, and Duhamel transport expansions
through third order.

The finite-difference-only two-level-system archive for the JCP3 manuscript
lives on the separate branch:

```text
codex/jcp3-finite-difference
```

## Scope

Included:

- `Liouville_autodiff.py`: optional JAX backend for Liouvillian connection
  tensors.
- `Liouville_duhamel.py`: Duhamel resolvent and propagator derivative
  expansions.
- `Liouville_transport_expansion.py`: model-agnostic transport-matrix Taylor
  expansion machinery.
- `exciton_dimer_model.py`: NumPy reference implementation of the four-level
  site-basis dimer.
- `exciton_dimer_autodiff.py`: JAX-compatible dimer Liouvillian.
- `demo_dimer_duhamel_transport.py`: production signal/transport workflow.
- `benchmark_dimer_transport_backends.py`: finite-difference versus
  autodiff/Duhamel timing and accuracy benchmark.
- `plot_dimer_backend_scaling.py`: benchmark scaling plots.
- `notebooks/01_model_and_signal_demo.ipynb`: dimer model and direct spectra.
- `notebooks/02_duhamel_transport_beta_analysis.ipynb`: autodiff/Duhamel
  transport workflow.
- `notebooks/03_custom_model_workflow.ipynb`: user-defined Liouvillian tutorial.
- `docs/autodiff_nodal_structure.md`: computational graph view of the autodiff
  and transport pipeline.

## Production Commands

Run the order-3 Duhamel/autodiff dimer production analysis:

```bash
python demo_dimer_duhamel_transport.py \
  --clean-reference-dimer \
  --clean-mu-esa-scale 1 \
  --n-omega 120 \
  --omega-min 0.8 \
  --omega-max 1.4 \
  --eta 0.003 \
  --backend duhamel_autodiff \
  --n-quad 4 \
  --order 3 \
  --sectors RP NRP
```

Run a compact backend benchmark:

```bash
python benchmark_dimer_transport_backends.py \
  --n-omega 12 16 24 \
  --n-quad 16 \
  --max-order 2
```

Run an order-3 Duhamel/autodiff benchmark:

```bash
python benchmark_dimer_transport_backends.py \
  --n-omega 120 \
  --n-quad 4 \
  --max-order 3 \
  --backends duhamel_autodiff
```

Regenerate timing/error scaling plots from existing benchmark summaries:

```bash
python plot_dimer_backend_scaling.py \
  dimer_backend_benchmarks/production_compare_n12_n16_n24/benchmark_summary.csv \
  dimer_backend_benchmarks/duhamel_autodiff_production_n40_n80_n120/benchmark_summary.csv \
  dimer_backend_benchmarks/20260722_163259_duhamel_autodiff_order3_n120_q4/benchmark_summary.csv \
  --output-dir dimer_backend_benchmarks/scaling_plots_autodiff_vs_fd
```

## Reference Production Result

The reference order-3 production run used:

- `n_omega = 120`
- `n_quad = 4`
- `omega in [0.8, 1.4] eV`
- `eta = 0.003`
- clean reference dimer with `J_12 = 0`, `k_12 = 0`
- target lambda `(-0.25, 0.25)`
- controls `J_12(lambda_J) = 0 + 0.045 lambda_J`
- controls `k_12(lambda_k) = 0 + 0.012 lambda_k`
- sectors `RP`, `NRP` transported independently

Representative diagnostics from that run:

- exact projected signal error: `2.859731958155e-01`
- first-order Duhamel signal error: `2.883142283939e-01`
- second-order Duhamel signal error: `2.867020208725e-01`
- third-order Duhamel signal error: `2.858650081202e-01`
- fitted transport signal error: `2.768373204723e-01`
- `||T_order3|| / ||T_exact_projected|| = 1.000840922184`
- order-3 transport-matrix relative error: `2.11575e-03`

These numbers are mainly validation diagnostics. The conceptual result is that
the spectral features remain fairly stable while the transport matrix exposes
how reference pathways mix under the distorted Liouvillian.

## Validation Tests

Run the focused SAKE/autodiff test set:

```bash
python -m pytest -q \
  test_exciton_dimer_autodiff.py \
  test_liouville_autodiff.py \
  test_liouville_duhamel.py \
  test_exciton_dimer_model.py \
  test_v_system_general.py \
  test_transport_expansion.py \
  test_liouville_transport_general.py \
  test_phase_matched_response.py
```

The last full focused test pass before the branch split was:

```text
41 passed, 1 skipped
```

## Notes

- The branch keeps generated production plots and large run directories ignored
  by git. Recreate them with the commands above.
- JAX is optional at import time, but required for `duhamel_autodiff` runs.
- The transport expansion is model agnostic once the Liouvillian derivative
  dictionary has been constructed.
