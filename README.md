# Duhamel Transport

Liouville-space pathway transport code for the rotated-pointer-basis two-level
model used in the nonlinear-response/Duhamel reconstruction workflow.

## Publication Branch: JCP3 Finite Difference

This branch is the finite-difference-only archive for the two-level system plus
rotated dephasing model used in the JCP3 manuscript under review. It does not
include the autodiff/dimer development line.

The curated validation assets for the paper branch are in
`publication_assets/jcp3_finite_difference/`. They were regenerated from the
commands in `docs/jcp3_finite_difference_release.md`.

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

## Notes

Generated plots and diagnostic output directories are ignored by git. Recreate
them with the run scripts above.
