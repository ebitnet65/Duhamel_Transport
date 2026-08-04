# JCP3 Finite-Difference Release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21793116.svg)](https://doi.org/10.5281/zenodo.21793116)

This branch archives the finite-difference-only implementation of
Liouville-space pathway transport for the rotated two-level-system plus
dephasing model used in the JCP3 manuscript.

The branch is intentionally scoped to the JCP3 development. The later
autodiff, four-level exciton-dimer, and third-order Duhamel work live on the
separate SAKE/autodiff branch.

## Citation

Release DOI:

```text
10.5281/zenodo.21793116
```

DOI link: <https://doi.org/10.5281/zenodo.21793116>

Suggested citation:

```text
Bittner, E. R. Duhamel Transport v0.1: Finite-Difference Two-Level-System
Release for JCP3. Zenodo. https://doi.org/10.5281/zenodo.21793116
```

## Contents

Core files:

- `phase_matched_response.py`: rotated-pointer-basis two-level Liouvillian and
  phase-matched RWA response utilities.
- `Liouville_transport.py`: finite-lambda pathway construction, reference-basis
  projection, and transport-matrix diagnostics.
- `run_liouville_transport.py`: finite-difference transport sweep and plotting
  driver.

Diagnostics:

- `diagnose_reference_basis.py`: reference pathway-basis conditioning.
- `diagnose_transport_stages.py`: stage-by-stage pathway conditioning.
- `test_phase_matched_response.py`: algebraic smoke check comparing
  `COMM_PLUS` and `ALL_RWA` response constructions.

Not included in this branch:

- JAX/autodiff Liouvillian derivative code.
- Four-level exciton-dimer model and third-order Duhamel production workflow.
- SAKE release-development notebooks.

## Regenerating Validation Outputs

Use a non-interactive matplotlib backend when regenerating plots from a terminal
or remote session.

Algebraic response check:

```bash
python test_phase_matched_response.py
```

Transport matrices and residual diagnostics:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python run_liouville_transport.py \
    --n-omega 101 \
    --n-lambda 21 \
    --output-dir publication_assets/jcp3_finite_difference/transport_output
```

Reference-basis diagnostics:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python diagnose_reference_basis.py \
    --n-omega 101 \
    --output-dir publication_assets/jcp3_finite_difference/basis_diagnostics
```

Stage diagnostics:

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-duhamel \
  python diagnose_transport_stages.py \
    --n-omega 101 \
    --output-dir publication_assets/jcp3_finite_difference/stage_diagnostics
```

## Archived Validation Run

The tracked summaries and compact numerical data were generated with:

| Quantity | Value |
| --- | --- |
| `n_omega` | `101` |
| `n_lambda` | `21` |
| `theta` range | `[0, pi/2]` |
| `omega` range | `[0.25, 1.75]` |
| `gamma_phi` | `0.1` |
| `gamma1` | `0.02` |
| `eta` | `1e-3` |
| `tau2` | `1.0` |
| initial state | `fixed_steady0` |

Representative checks:

| Check | Value |
| --- | --- |
| `COMM_PLUS` vs `ALL_RWA` relative mismatch | `9.568528191617517e-18` |
| RP `||T(0)-I||` | `6.342482549767e-16` |
| NRP `||T(0)-I||` | `9.339823478300e-16` |
| RP retained basis at `after_U2` | `RP-GSB`, `RP-SE` |
| NRP retained basis at `after_U2` | `NRP-R+L-L+`, `NRP-R+R-L+` |
| RP/NRP rank at `after_U2` | `2` at relative tolerance `1e-12` |

## Validation Assets

Tracked release assets live under:

```text
publication_assets/jcp3_finite_difference/
```

The branch tracks text summaries and compact `.npz` numerical outputs. PNG
plots are generated artifacts and are ignored by git; regenerate them with the
commands above.

Useful generated plots include:

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
