# Dark-state trap-rate stress test

**Date:** 2026-10-22  
**Branch:** `codex/spectral-sensitivity-model-discrimination`

## Question

How far can the original three-parameter, sink-free exciton dimer absorb
spectral changes caused by a physically real but optically dark terminal trap?
This is a controlled model stress test. The synthetic truth includes the trap;
the fitted candidate does not.

## Models and calculation

The constant dimer truth parameters are
`(J12, k12, gamma_phi) = (0.025, 0.014, 0.022)`. The truth Liouvillian adds
`L_d = sqrt(kd) |d><e1|` for trap rates
`kd = 0.003, 0.01, 0.03, 0.1, 0.3, 1.0`. The dark state has no optical dipole,
coherent coupling, or return pathway. Its population remains inert during
subsequent pulses. The candidate is the unchanged three-control dimer without
the sink.

For each rate, the calculation uses the same complex RP+NRP observable, a
21-by-21 frequency grid from 0.90 to 1.28 eV, `eta = 0.035`, and population
waiting times `t2 = 0.2, 1, 4`. The candidate is represented by a third-order
autodiff/Duhamel transport expansion with `n_quad = 2` and a five-point
projection grid. It is fit jointly across all three waiting times with one
global parameter triplet; independent single-waiting-time fits are also saved.
The objective is the unweighted sum of squared real and imaginary spectral
residuals, with the same scaled-control bounds in every case.

## Results

| `kd` | Joint fit `(J12, k12, gamma_phi)` | Relative complex-spectrum L2 error at `t2 = 0.2, 1, 4` |
|---:|---|---|
| 0.003 | (0.02558, 0.01240, 0.02256) | 0.72%, 0.72%, 0.77% |
| 0.01 | (0.02689, 0.00880, 0.02383) | 2.31%, 2.30%, 2.43% |
| 0.03 | (0.03062, 0.00200, 0.02642) | 6.18%, 6.14%, 6.23% |
| 0.1 | (0.04268, 0.00200, 0.02824) | 14.71%, 14.42%, 13.34% |
| 0.3 | (0.06023, 0.00200, 0.03247) | 22.70%, 21.72%, 18.75% |
| 1.0 | (0.06500, 0.00728, 0.04000) | 30.37%, 28.20%, 25.97% |

The three-control model first shows a conspicuous stress at `kd = 0.03`:
joint `k12` reaches its lower bound (`0.002`) and the relative spectral error
is about 6%. At `kd = 0.1`, all independent fits also reach the `k12` lower
bound while `J12` and `gamma_phi` shift from their fixed truth values. At
`kd = 0.3`, every fit is at the `k12` lower bound and the residual is 19–23%.
At `kd = 1.0`, the joint fit reaches the imposed upper bounds for `J12` and
`gamma_phi` (`0.065` and `0.040`); `k12` moves back into the interior, but
the residual remains 26–30%. That rebound reflects parameter compensation
near the bounds, not restoration of a credible parameterization.

The fit code reports optimizer success for the joint runs, but boundary hits
and the large structured residuals show why optimizer success alone is not
evidence that the candidate is adequate. Full per-waiting-time estimates and
errors are recorded in
[`dark_state_trap_sweep_metrics.csv`](../publication_artifacts/dark_state_transport/trap_sweep/dark_state_trap_sweep_metrics.csv)
and the machine-readable
[`summary JSON`](../publication_artifacts/dark_state_transport/trap_sweep/dark_state_trap_sweep_summary.json).

## Validation and interpretation

All numerical arrays are finite. For each trap rate, the full trace-preserving
sink model and the independently evaluated active-block response agree with
relative errors between `2.56e-15` and `3.01e-15`. The first-order conclusion
is therefore a model-adequacy result under these specified conventions: the
sink-free fit increasingly fails as trap loss grows, and the optimizer
compensates by pushing available controls to their bounds.

These runs are deterministic and noiseless. They include no uncertainty
intervals, measurement noise, or formal significance test. Residual size here
does not define an experimental detection threshold, and the fit cannot
establish that the omitted microscopic mechanism in real data is specifically
this trap. Establishing microscopic identity still requires comparing
physically motivated extended models against noisy, held-out observations.

## Figures and reproduction

- [Parameter and residual trends](../publication_artifacts/dark_state_transport/trap_sweep/figure_dark_state_trap_rate_sweep.pdf)
- [Largest relative residual spectrum (`kd=1`, `t2=0.2`)](../publication_artifacts/dark_state_transport/trap_sweep/figure_dark_state_strong_trap_residuals.pdf)
- [Per-rate dark-truth versus transported-candidate figures and spectra](../publication_artifacts/dark_state_transport/trap_sweep/)
- [Model and optical-convention documentation](../publication_artifacts/dark_state_transport/README.md)

Reproduce the aggregate table and summary figures after the individual
directories exist:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_trap_sweep.py
```

For example, regenerate an individual `kd = 0.3` case with:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_transport_figures.py \
  --output-dir publication_artifacts/dark_state_transport/trap_sweep/kd_0.3 \
  --nfreq 21 --projection-nfreq 5 --waiting-times 0.2 1.0 4.0 \
  --n-quad 2 --order 3 --trap-rate 0.3
```
