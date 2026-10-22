# Dark-state truth versus Duhamel-transport dimer

This production figure uses the requested example: the synthetic truth is an
exciton dimer with an optically dark terminal sink, while the fitted candidate
is the original three-parameter, sink-free dimer. “Transported” means the
candidate spectrum is reconstructed from its reference RP/NRP pathway basis by
the third-order autodiff/Duhamel expansion; it is not a direct spectrum from a
separately optimized exact propagator.

## Reproduction

From the repository root, run:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_transport_figures.py \
  --output-dir publication_artifacts/dark_state_transport \
  --nfreq 21 --projection-nfreq 5 --waiting-times 0.2 1.0 4.0 \
  --n-quad 2 --order 3 --trap-rate 0.003
```

The production driver is headless and deterministic. It writes two vector PDF
figures, 600 dpi PNG previews, `dark_state_transport_comparison.npz` with the
complex spectra and fit arrays, and `dark_state_transport_comparison.json`
with configuration, fit results, and numerical validation. The PNG/PDF files
are generated from the saved arrays in the same run.

## Model and optical convention

The fixed dimer truth parameters are
`(J12, k12, gamma_phi) = (0.025, 0.014, 0.022)` in the repository's energy and
inverse-time units. The truth trap rate is `kd = 0.003`. The five-state
Hilbert-space ordering is ground, two one-exciton site states, the existing
double-excitation state, and a terminal sink. The sink has no Hamiltonian
coupling, optical dipole, return route, or optically accessible excitation
manifold. The only added collapse operator is
`sqrt(kd) |d><e1|`. It transfers population to the sink and attenuates
coherences involving `|e1>` at half that rate. During later pulses, sink
population remains inert and creates no optical transition.

All RP and NRP canonical pathways are included for the truth and candidate.
The reduced candidate basis is selected independently by sector using the
repository's canonical pathway rank-reduction utility. At the reference
parameters, a single complex weight vector across the combined RP/NRP retained
basis is calibrated to reproduce the full canonical RP+NRP sum on the display
grid. The transported model then varies the three physical controls through a
third-order Duhamel expansion. The plotted candidate is a joint fit with one
parameter triplet across all three waiting times; the parameterization figure
also shows independent fits for each time.

The active-block spectrum is checked against the explicit five-state,
trace-preserving pathway response on a smaller frequency subset. The measured
relative discrepancy is `3.15e-15`. Reference-basis closure errors on the
production grid are `9.30e-15`, `9.39e-16`, and `7.61e-16` at `t2 = 0.2, 1,
4`, respectively. The cached pathway evaluator used for dense-grid production
was separately compared with the repository pointwise helper; their relative
difference was `1.43e-16`.

## Results and limits

The joint sink-free transported fit is
`(0.025583, 0.012402, 0.022563)`. Its relative complex-spectrum L2 errors
against truth are `0.00722`, `0.00723`, and `0.00770` at the three waiting
times. Independent fits are:

| `t2` | `J12` | `k12` | `gamma_phi` | relative complex L2 error |
|---:|---:|---:|---:|---:|
| 0.2 | 0.024797 | 0.002000 | 0.025424 | 0.00632 |
| 1.0 | 0.024837 | 0.005652 | 0.024457 | 0.00664 |
| 4.0 | 0.025864 | 0.013805 | 0.022158 | 0.00748 |

These are deterministic, noiseless fits. No confidence intervals or
noise-based significance claim is implied. In particular, the `t2=0.2`
independent `k12` estimate reaches the imposed scaled-control lower bound
(`0.002`); its displacement is therefore not evidence by itself of a
well-identified physical drift. The joint fit is a single best parameterization
under this basis, order, grid, and objective. The residuals show that a
sink-free model cannot exactly reproduce the dark-trap truth, while the good
fit does not establish the microscopic identity of an omitted mechanism.

Interpret the comparison as a controlled model-stress figure: it demonstrates
how a missing dark channel can be partly absorbed into fitted dimer parameters
and leaves a structured complex residual. It does not replace the larger
noise-aware, uncertainty-quantified structural-discrimination benchmark in
`docs/`.

## Trap-rate stress sweep

The checked-in `kd=0.003` outputs are retained as the fixed reference case.
Higher-rate runs are written to separate directories under
`trap_sweep/kd_<rate>` so they cannot overwrite the reference. Each run uses
the same frequency grid, waiting times, control bounds, and third-order
transport calculation; only `kd` changes. Reproduce an individual case with:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_transport_figures.py \
  --output-dir publication_artifacts/dark_state_transport/trap_sweep/kd_0.01 \
  --nfreq 21 --projection-nfreq 5 --waiting-times 0.2 1.0 4.0 \
  --n-quad 2 --order 3 --trap-rate 0.01
```

After the case directories are populated, aggregate all completed rates with:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_trap_sweep.py
```

The aggregator preserves the original baseline directory, writes a CSV/JSON
table of fitted parameters and relative complex-spectrum errors, and generates
publication figures for parameter drift, residual growth, and the strongest
completed trap case. The sweep interpretation remains conditional on the
specified spectral window, grid, control bounds, and noiseless least-squares
objective; it is not an uncertainty or mechanism-identification result.

The completed sweep (`kd = 0.003, 0.01, 0.03, 0.1, 0.3, 1.0`) shows increasing
model mismatch:

| `kd` | joint `(J12, k12, gamma_phi)` | joint relative complex L2 error over `t2 = 0.2, 1, 4` |
|---:|---|---|
| 0.003 | (0.02558, 0.01240, 0.02256) | 0.72%, 0.72%, 0.77% |
| 0.01 | (0.02689, 0.00880, 0.02383) | 2.31%, 2.30%, 2.43% |
| 0.03 | (0.03062, 0.00200, 0.02642) | 6.18%, 6.14%, 6.23% |
| 0.1 | (0.04268, 0.00200, 0.02824) | 14.71%, 14.42%, 13.34% |
| 0.3 | (0.06023, 0.00200, 0.03247) | 22.70%, 21.72%, 18.75% |
| 1.0 | (0.06500, 0.00728, 0.04000) | 30.37%, 28.20%, 25.97% |

At `kd = 0.03`, the joint fit drives `k12` to its imposed lower bound of
`0.002`; at `kd = 0.1`, all independent fits do so as well. `J12` and
`gamma_phi` move away from their constant truth values. At `kd = 0.3`, the
joint and all independent `k12` estimates remain at the lower bound. At
`kd = 1.0`, the joint fit instead reaches the imposed upper bounds on `J12`
and `gamma_phi`; joint `k12` moves back into the interior, but residuals remain
above 25% at every waiting time. This rebound is a boundary/compensation
effect, not restoration of a credible dimer parameterization. These noiseless
residuals establish model stress in this controlled setting but do not imply
experimental significance or identify the missing mechanism.
