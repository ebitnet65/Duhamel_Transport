# SAKE-DT Autodiff Dimer Release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21793438.svg)](https://doi.org/10.5281/zenodo.21793438)

**SAKE-DT** stands for:

```text
Spectral Autodiff Kernel Expansion for Duhamel Transport
```

This branch is the release-development line for the SAKE-DT code: a
JAX/autodiff and Duhamel-expansion implementation of Liouville pathway
transport for a four-level `su(2) x su(2)` exciton dimer.

The finite-difference-only two-level-system archive for the JCP3 manuscript
lives on:

```text
codex/jcp3-finite-difference
```

## Citation

Release DOI:

```text
10.5281/zenodo.21793438
```

DOI link: <https://doi.org/10.5281/zenodo.21793438>

Suggested citation:

```text
Bittner, E. R. SAKE-DT v0.1: Spectral Autodiff Kernel Expansion for
Duhamel Transport. Zenodo. https://doi.org/10.5281/zenodo.21793438
```

## Suggested Release Metadata

Suggested GitHub tag:

```text
v0.1-sake-autodiff-dimer
```

Suggested release title:

```text
SAKE-DT v0.1: Spectral Autodiff Kernel Expansion for Duhamel Transport
```

Suggested short description:

```text
Beta release of SAKE-DT, a Spectral Autodiff Kernel Expansion framework for
Duhamel transport of Liouville-space nonlinear-response pathways.
```

Suggested keywords:

```text
Duhamel expansion; Liouville space; autodiff; JAX; nonlinear spectroscopy;
two-dimensional spectroscopy; exciton dimer; pathway transport; response
functions; spectral reconstruction
```

## Scope

Core transport and derivative machinery:

- `Liouville_autodiff.py`: JAX backend for Liouvillian connection tensors.
- `Liouville_duhamel.py`: Duhamel resolvent and propagator derivative
  expansions.
- `Liouville_transport_general.py`: model-agnostic Liouville transport API.
- `Liouville_transport_expansion.py`: Taylor expansion of the projected
  pathway transport matrix `T(lambda)`.

Dimer model and production scripts:

- `exciton_dimer_model.py`: NumPy reference implementation of the four-level
  site-basis dimer.
- `exciton_dimer_autodiff.py`: JAX-compatible dimer Liouvillian.
- `demo_dimer_duhamel_transport.py`: order-1/order-2/order-3 transport and
  signal production workflow.
- `benchmark_dimer_transport_backends.py`: finite-difference versus
  autodiff/Duhamel timing and accuracy benchmark.
- `plot_dimer_backend_scaling.py`: benchmark scaling plots.

Tutorial and documentation material:

- `notebooks/01_model_and_signal_demo.ipynb`: dimer model construction and
  direct reference/target spectra.
- `notebooks/02_duhamel_transport_beta_analysis.ipynb`: autodiff/Duhamel
  transport matrix workflow.
- `notebooks/03_custom_model_workflow.ipynb`: user-defined Liouvillian
  tutorial.
- `docs/autodiff_nodal_structure.md`: computational graph view of the autodiff
  and transport pipeline.

## Central Object

SAKE-DT transports reference pathway states rather than recomputing spectra
from scratch at every parameter point:

```text
P_a(lambda) ~= sum_b T_a^b(lambda) P_b(0)
```

The final observable signal is formed only after transport, by summing the
transported pathway signals or tracing the transported density object against
the detection operator.

The selected controls are dimensionless displacements from a reference model:

```text
p_mu(lambda_mu) = p_mu,ref + delta_mu lambda_mu
```

For the beta dimer release, the default controls are:

```text
J_12(lambda_J) = 0 + 0.045 lambda_J
k_12(lambda_k) = 0 + 0.012 lambda_k
```

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

## Reference Production Run

The reference order-3 production run used:

| Quantity | Value |
| --- | --- |
| `n_omega` | `120` |
| `n_quad` | `4` |
| `omega` range | `[0.8, 1.4] eV` |
| `eta` | `0.003` |
| reference model | clean dimer with `J_12 = 0`, `k_12 = 0` |
| target lambda | `(-0.25, 0.25)` |
| phase matching | RP and NRP transported independently |

Representative diagnostics:

| Check | Value |
| --- | --- |
| exact projected signal error | `2.859731958155e-01` |
| first-order Duhamel signal error | `2.883142283939e-01` |
| second-order Duhamel signal error | `2.867020208725e-01` |
| third-order Duhamel signal error | `2.858650081202e-01` |
| fitted transport signal error | `2.768373204723e-01` |
| `||T_order3|| / ||T_exact_projected||` | `1.000840922184` |
| order-3 transport-matrix relative error | `2.11575e-03` |

The spectral features remain fairly stable across this local displacement. The
transport matrix is the more diagnostic object: it exposes how reference
pathway weights mix under the distorted Liouvillian.

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

Current branch validation:

```text
41 passed, 1 skipped
```

## Notes

- Generated production plots and large run directories are ignored by git.
  Recreate them with the commands above.
- JAX is optional at import time, but required for `duhamel_autodiff` runs.
- Once the Liouvillian derivative dictionary is constructed, the Duhamel
  transport expansion is model agnostic.
- The finite-difference backend remains available as a reference implementation
  for validating autodiff/Duhamel derivatives.
