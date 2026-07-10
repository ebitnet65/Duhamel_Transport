# Duhamel Transport

Liouville-space pathway transport code for the rotated-pointer-basis two-level
model used in the nonlinear-response/Duhamel reconstruction workflow.

The central calculation projects finite-parameter pathway states onto a
canonical lambda-zero Liouville pathway basis:

```text
|P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>
```

The resulting transport matrices `T_ab(lambda)` quantify how phase-matched
pathway components deform under the rotated dephasing Liouvillian.

## Main Files

- `phase_matched_response.py`: phase-matched RP/NRP two-level response model.
- `Liouville_transport.py`: pathway-state construction, basis projection, and
  transport-matrix diagnostics.
- `Liouville_transport_general.py`: model-agnostic transport core where users
  supply a Liouvillian, initial state, transition dipole operator, and selected
  control variables.
- `run_liouville_transport.py`: command-line driver for transport sweeps and
  plots.
- `Compute_responses.py`: response computation helper.
- `curvature_plots.py`: plotting utilities for response/curvature outputs.
- `diagnose_reference_basis.py`: lambda-zero basis conditioning diagnostics.
- `diagnose_transport_stages.py`: stage-by-stage basis conditioning diagnostics.
- `run_all_diagnostics.sh`: runs both diagnostic scripts.

## Basic Run

From this directory:

```bash
python run_liouville_transport.py
```

Use a denser theta sweep:

```bash
python run_liouville_transport.py --n-lambda 101
```

Use the instantaneous steady state instead of the fixed reference state:

```bash
python run_liouville_transport.py --initial steady
```

Choose another output directory:

```bash
python run_liouville_transport.py --output-dir results_transport
```

## Diagnostics

Run the reference-basis diagnostic:

```bash
python diagnose_reference_basis.py
```

Run the stage comparison:

```bash
python diagnose_transport_stages.py
```

Or run both:

```bash
bash run_all_diagnostics.sh
```

Key quantities:

- `||T(0)-I||`: should be near numerical precision.
- `rank`: should equal the number of retained canonical pathways.
- `condition number`: large values indicate a nearly dependent pathway basis.
- `projection residuals`: quantify how much of each finite-lambda pathway lies
  outside the canonical lambda-zero pathway subspace.

## General Transport Core

`Liouville_transport_general.py` separates full model parameters from the
selected lambda-vector controls. For example, a model may require `w0`,
`gamma_phi`, `gamma1`, and `theta`, while the expansion is performed only with
respect to `theta` or only with respect to `(theta, gamma_phi)`.

The user-facing pieces are:

- `LiouvilleModel`: supplies `L(params)`, `rho0(params)`, and the transition
  dipole `mu(params)`.
- `ParameterMap`: maps a lambda vector into the full parameter dictionary.
- `ControlSpec`: names each selected control coordinate and its scale.
- `canonical_pathways`: builds nested-commutator RWA pathways.
- `liouville_transport_matrix`: computes `T(lambda)` by projecting finite
  pathway states onto the lambda-zero pathway basis.

The current derivative helper,
`finite_difference_liouvillian_derivatives`, fixes the indexing convention for
first and second derivatives with respect to selected controls. It is intended
as the reference backend before adding autodiff.

## Notes

Generated plots and diagnostic output directories are ignored by git. Recreate
them with the run scripts above.
