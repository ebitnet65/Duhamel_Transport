# JCP3 Finite-Difference Release Branch

This branch archives the finite-difference-only code path for the rotated
two-level-system plus dephasing model used in the JCP3 manuscript.

It intentionally excludes the later autodiff and exciton-dimer development.
Those features live on the separate SAKE/autodiff branch.

## Scope

Included:

- `phase_matched_response.py`: rotated-pointer-basis two-level Liouvillian and
  phase-matched RWA response utilities.
- `Liouville_transport.py`: finite-lambda pathway construction, reference-basis
  projection, and transport-matrix diagnostics.
- `run_liouville_transport.py`: finite-difference transport sweep and plotting
  driver.
- `diagnose_reference_basis.py`: reference pathway-basis conditioning
  diagnostic.
- `diagnose_transport_stages.py`: stage-by-stage conditioning diagnostic.
- `test_phase_matched_response.py`: algebraic smoke check comparing
  `COMM_PLUS` and `ALL_RWA` response constructions.

Excluded:

- JAX/autodiff Liouvillian derivative code.
- Four-level exciton-dimer and third-order Duhamel production code.
- SAKE release-development notebooks.

## Validation Commands

Run the algebraic response smoke check:

```bash
python test_phase_matched_response.py
```

Regenerate the curated finite-difference transport plots:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python run_liouville_transport.py \
    --n-omega 101 \
    --n-lambda 21 \
    --output-dir publication_assets/jcp3_finite_difference/transport_output
```

Regenerate the reference-basis diagnostics:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python diagnose_reference_basis.py \
    --n-omega 101 \
    --output-dir publication_assets/jcp3_finite_difference/basis_diagnostics
```

Regenerate the stage diagnostics:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python diagnose_transport_stages.py \
    --n-omega 101 \
    --output-dir publication_assets/jcp3_finite_difference/stage_diagnostics
```

## Validation Summary

The archived validation run used:

- `n_omega = 101`
- `n_lambda = 21`
- `theta in [0, pi/2]`
- `omega in [0.25, 1.75]`
- `gamma_phi = 0.1`
- `gamma1 = 0.02`
- `eta = 1e-3`
- `tau2 = 1.0`
- initial state `fixed_steady0`

Representative checks from the run:

- `COMM_PLUS` versus `ALL_RWA` relative mismatch:
  `9.568528191617517e-18`.
- RP transport identity check:
  `||T(0)-I|| = 6.342482549767e-16`.
- NRP transport identity check:
  `||T(0)-I|| = 9.339823478300e-16`.
- Retained RP basis at `after_U2`: `RP-GSB`, `RP-SE`.
- Retained NRP basis at `after_U2`: `NRP-R+L-L+`, `NRP-R+R-L+`.
- RP and NRP basis rank at `after_U2`: `2` at relative tolerance `1e-12`.

## Curated Assets

The branch includes tracked text summaries and compact numerical outputs under:

```text
publication_assets/jcp3_finite_difference/
```

The validation commands above regenerate the manuscript/SI plots in the same
directory. The most useful generated plots for manuscript or SI reference are:

- `transport_output/RP/RP_T_abs_final.png`
- `transport_output/RP/RP_T_real_final.png`
- `transport_output/RP/RP_T_elements_abs.png`
- `transport_output/RP/RP_projection_residuals.png`
- `transport_output/NRP/NRP_T_abs_final.png`
- `transport_output/NRP/NRP_T_real_final.png`
- `transport_output/NRP/NRP_T_elements_abs.png`
- `transport_output/NRP/NRP_projection_residuals.png`
- `basis_diagnostics/RP_gram_abs.png`
- `basis_diagnostics/NRP_gram_abs.png`
- `stage_diagnostics/RP_condition_by_stage.png`
- `stage_diagnostics/NRP_condition_by_stage.png`

The `.npz` files are included so selected transport plots can be regenerated or
reanalyzed without rerunning the full transport sweep. The PNG plots themselves
are treated as generated artifacts and are ignored by git.
