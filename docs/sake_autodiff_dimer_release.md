# SAKE-DT Autodiff Dimer v0.2 Release

[![previous release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21793438.svg)](https://doi.org/10.5281/zenodo.21793438)

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

Version-specific v0.2 DOI:

```text
[reserved in the Zenodo v0.2 draft]
```

Previous v0.1 DOI: <https://doi.org/10.5281/zenodo.21793438>

Suggested citation:

```text
Bittner, E. R. SAKE-DT v0.2: Spectral Autodiff Kernel Expansion for
Duhamel Transport. Zenodo. [version DOI]
```

## Suggested Release Metadata

Suggested GitHub tag:

```text
v0.2-sake-autodiff-dimer
```

Suggested release title:

```text
SAKE-DT v0.2: Spectral Autodiff Kernel Expansion for Duhamel Transport
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

### Systematic accuracy and amortized timing

The systematic benchmark uses explicit target-model resolvent inversions as
the primary ground truth. It separates reference-basis projection error from
Taylor/Duhamel truncation error and total observable-signal error:

The reference dimer is deliberately asymmetric,
$\omega_1=\omega_0+\Delta\omega/2$ and
$\omega_2=\omega_0-\Delta\omega/2$. Both controls are nondimensionalized by
the mean site energy, so $\lambda_J=J/\omega_0$,
$\lambda_\kappa=\kappa/\omega_0$, and
$\lVert\boldsymbol{\lambda}\rVert=\sqrt{J^2+\kappa^2}/\omega_0$.
The defaults use $\omega_0=2.0$ eV and $\Delta\omega=0.18$ eV, giving site
energies of 2.09 and 1.91 eV. The spectral window follows $\omega_0$ unless
`--omega-min` and `--omega-max` are supplied.

```bash
python prepare_sake_dimer_derivatives.py \
  --omega0 2.0 \
  --delta-omega 0.18 \
  --max-order 2 \
  --output dimer_backend_benchmarks/sake_dimer_derivatives.npz

python benchmark_sake_accuracy_timing.py \
  --derivative-cache dimer_backend_benchmarks/sake_dimer_derivatives.npz \
  --omega0 2.0 \
  --delta-omega 0.18 \
  --n-omega 16 \
  --radii 0 0.0025 0.005 0.01 0.02 0.03 0.05 \
  --target-counts 1 2 4 8 16 32 64 \
  --timing-repeats 3 \
  --n-quad 4 \
  --output-dir dimer_backend_benchmarks/systematic_accuracy_timing_n16
```

The separate derivative-cache step keeps the memory used by JAX/XLA
compilation out of the longer pathway benchmark while retaining the measured
derivative-construction time in the amortized setup cost. For the clean dimer,
the Liouvillian is affine in the two selected controls, so second- and
third-order Liouvillian derivatives vanish. First derivatives are sufficient
mathematically, but `--max-order 2` is the supported minimum because it supplies
the explicit zero Hessian blocks required by the transport-builder interface.

Generated outputs include:

- `accuracy_vs_displacement.csv`, `.png`, and `.pdf`;
- `wall_time_vs_targets.csv`, `.png`, and `.pdf`;
- `benchmark_table.csv` and a REVTeX-ready `benchmark_table.tex`;
- `benchmark_metadata.json`, including control scales, basis conditioning,
  setup times, and the observed amortization break-even point.

To test the local radius of reliability around a nonzero physical reference,
prepare a matching derivative cache and pass the same reference controls to
the benchmark. For example, a midpoint chart between the uncoupled model and
the target $J=-0.100$ eV, $\kappa=0.012$ eV uses
`--reference-J -0.050 --reference-k 0.006`. The displacement columns in the
accuracy table remain relative to that reference, while `target_J` and
`target_k` report the absolute physical parameters.

### Iterative inverse recovery

The trust-region inverse utility uses a direct target calculation as synthetic
data, minimizes third-order SAKE pathway-state residuals inside successive
local charts, and validates each proposed chart center with one direct
inversion:

```bash
python fit_sake_dimer_iterative.py \
  --derivative-cache dimer_backend_benchmarks/sake_dimer_derivatives.npz \
  --target-J -0.100 \
  --target-k 0.012 \
  --initial-radius 0.005 \
  --maximum-radius 0.010 \
  --output-dir dimer_backend_benchmarks/iterative_inverse_target
```

The stable `accumulated_transport` output is the global projection from the
initial zeroth-order pathways to the recovered controls. The utility also saves
the naïve product of accepted local overlap maps as `composed_local_transport`.
Their difference is an explicit closure diagnostic: projected finite pathway
bases are not generally transitive when out-of-basis residuals are appreciable.
The output also contains the iteration history, direct-validation count, the
directly projected target counterpart, and composition diagnostics.

Multiple recovery directories can be combined into a two-panel,
publication-ready trajectory/convergence figure and manuscript table with
`plot_sake_multi_target_recovery.py`. Solid curves show direct pathway-state
validation and dashed curves show the local SAKE prediction; colors identify
the target throughout both panels.

The reported error measures are:

- `state_projection_error`: direct target pathway states versus their
  least-squares reconstruction in the reference pathway basis;
- `signal_projection_error`: directly inverted target signal versus the signal
  reconstructed with the directly projected transport matrix;
- `orderN_transport_error`: order-N SAKE transport matrix versus the directly
  projected target transport matrix;
- `orderN_signal_truncation_error`: order-N signal versus the signal obtained
  with the directly projected target matrix;
- `orderN_signal_total_error`: order-N signal versus direct target-model
  inversion.

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
