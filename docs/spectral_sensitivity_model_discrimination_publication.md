# Spectral sensitivity and hidden-channel stress test

## Purpose and scope

This report documents a small, reproducible synthetic benchmark for spectral
sensitivity, waiting-time information, and discrimination of a hidden terminal
trap from changes to an exciton-dimer model. The objective is to stress the
model and its inference workflow: determine when a reduced model absorbs a
missing channel as parameter drift, when an added dephasing term can mimic the
missing channel, and when a residual or additional measurement provides
evidence against that reduced description.

The calculations are diagnostics on synthetic spectra, not a converged
publication-scale spectroscopy study and not evidence about an experimental
sample. In particular, fit inconsistency, improved prediction after adding a
mechanism, and identification of that mechanism's microscopic identity are
three different conclusions. The data here can illustrate the first two
conditionally; they cannot establish the third.

## Model and optical conventions

The preserved benchmark is the four-state site-product dimer with basis
`|g>, |e1>, |e2>, |f>`: ground, two one-exciton site states, and the doubly
excited state. Its three fitted controls are
`(J12, k12, gamma_phi) = (0.025, 0.014, 0.022)`. The site-energy gap,
temperature, lifetime rates, dipoles, and double-excitation anharmonicity
remain fixed. The reference observable and its tests are unchanged.

The optical response follows the existing RWA nested-commutator implementation
in `dimer_spectral_observable.py`. The dipole includes ground-to-single
transitions and single-to-double transitions scaled by `mu_esa_scale`.
Rephasing uses field signs `(-,+,+)` and nonrephasing uses `(+,-,+)` under the
repository's field convention. The first and third coherence intervals use
the existing Liouvillian resolvents; the population interval uses the
Liouvillian exponential. RP, NRP, and RP+NRP spectra remain complex. Inference
stacks all real samples and then all imaginary samples; it does not silently
replace the measurement with absorptive-only data.

The structural extension adds a terminal state `|d>` and jump
`L_d = sqrt(k_d)|d><e1|`. It has no dipole, Hamiltonian coupling, or return
pathway. It is outside the optically accessible `g/e/f` manifolds. Optical
pulses therefore act only in the active four-state block, and any population
already in `|d>` remains there through subsequent pulses. No spurious
transitions to or from `|d>` are added.

This optical calculation is implemented using the exact active-block
projection of the full trace-preserving trap Lindbladian. On the active block,
the trap contributes

`L_trap rho = -(k_d/2){P1,rho}`, with `P1=|e1><e1|`.

The same generator is used for both frequency-domain coherence resolvents,
the waiting-time propagator, and the detection resolvent (the t1, t2, and t3
intervals). Thus the implementation does not model a sink only during t2.
For a full five-state density matrix, the added channel gives
`dot(rho_11)=-k_d rho_11`, `dot(rho_dd)=+k_d rho_11`, and
`dot(rho_12)=-(k_d/2)rho_12`; its total trace derivative is zero. The
four-state active block alone loses trace at rate `k_d rho_11`. Tests compare
this projected generator with the full sink generator and separately verify
that `sqrt(gamma_d)P1` preserves excited population while damping coherence.
The latter is genuine local pure dephasing, not a trap surrogate by
definition.

Candidate models use a common optical observable and weighting:

| Model | Generator extension | Fitted controls |
|---|---|---|
| A | Baseline dimer `L0(lambda)` | `J12, k12, gamma_phi` |
| B | `L0(lambda)+gamma_d D[P1]` | A controls and `gamma_d >= 0` |
| C | `L0(lambda)+k_d D[|d><e1|]` | A controls and `k_d >= 0` |

The A and zero-rate C observables are checked against the preserved baseline
to numerical precision.

## Sensitivity and information conventions

For complex spectrum `S(t2)`, the differentiable Jacobian is calculated by
JAX autodiff. The real inference matrix is `J_RI = [Re(J); Im(J)]`, matching
the stacked observation vector. A centered-difference calculation provides
an independent derivative check. Under independent Cartesian Gaussian noise
with standard deviation `sigma`, whitening is `W=I/sigma` and
`I=J_RI.T W.T W J_RI`. Singular values and the numerical rank describe local
parameter observability; the local covariance and parameter correlations are
computed from a pseudoinverse of this information matrix. These are
linearized diagnostics, not global uncertainty intervals.

Generalized modes use the fixed coordinate-scale metric
`G=diag(1/s_i^2)`, with reference scales
`s=(0.05 energy, 0.015 rate, 0.025 rate)` for `(J12,k12,gamma_phi)`.
The convention gives unit squared length to a displacement of one reference
scale along each coordinate and makes the reported modes dimensionless under
those selected scales. It is a normalization choice, not a uniquely derived
microscopic metric. It is held fixed while `I(t2)` changes with waiting time;
changes in information must not be interpreted as changes to the physical
metric. No claim of coordinate invariance removes the dependence of a
generalized mode on this metric choice.

For multiple waits, information is summed under equal per-sample noise weights.
Candidate pairs are ranked by their minimum information eigenvalue and log
determinant. For hidden-channel discrimination, the reduced-model tangent is
computed in whitened real/imaginary spectral space using an SVD basis. The
component of a trap perturbation outside the fitted tangent gives the part
that cannot be absorbed, to first order, by changing fitted parameters.
The extra-dephasing coefficient is constrained nonnegative in the local
projection.

## Numerical protocols and artifacts

The baseline sensitivity/model-discrimination smoke run uses frequencies
`[0.90, 1.09, 1.28]`, waits `[0.2, 1.0, 4.0]`, `eta=0.035`, and Cartesian
complex-component noise `sigma=0.001`. Trap rates are
`[0, 0.0003, 0.003, 0.012, 0.04]`. Its 3 by 3 frequency grid exercises the
pipeline but is too coarse to establish grid-converged spectral features.
Synthetic truth parameters are constant across all waits. A fits are performed
independently by waiting-time slice and jointly; the A/B/C fits use shared
truth spectra, weighting, and optical conventions. Held-out waits provide a
prediction check. The saved JSON contains the diagnostics and fit records;
the NPZ contains complex spectra and Jacobians.

The companion adversarial stress test uses a 5 by 5 grid from 0.90 to 1.28,
the same three waits, `eta=0.035`, trap rates 0, 0.0003, and 0.003, relative
noise levels 0.1%, 1%, and 5%, and 64 fixed seeds (1907–1970). Noise is iid
Gaussian on the real and imaginary samples, with sigma defined relative to the
measured-signal RMS. It compares early, late, early-plus-late, and all-wait
designs, and equal-budget nine-pixel center, low-frequency, high-frequency,
and geometry-selected cross-peak windows, as well as the full 25-pixel grid.
The comparison scores the tangent-space residual after A or B refits the
baseline tangent; B's local dephasing rate is nonnegative. The reported
Monte-Carlo frequency that B beats A's AIC-style score includes a two-unit
penalty for B's extra rate.

This stress test is deliberately adversarial and local. It does not run
nonlinear fits over every noise realization. The finite-rate perturbation is
first order in `k_d`; the signal-perturbation-to-baseline ratio is saved to
flag when this approximation may be strained. Its model-selection frequencies
are conditional on the tangent approximation, iid isotropic noise, selected
windows, and candidate set. They are not calibrated experimental likelihoods
or universal detection probabilities. Correlated instrument noise, nuisance
parameters, model discrepancy, and nonlinear/boundary profiling are not
included.

Machine-readable run records:

- [`spectral_sensitivity_benchmark.json`](benchmarks/spectral_sensitivity_model_discrimination/spectral_sensitivity_benchmark.json)
- [`spectral_sensitivity_benchmark.npz`](benchmarks/spectral_sensitivity_model_discrimination/spectral_sensitivity_benchmark.npz)
- [`spectral_stress_test.json`](benchmarks/spectral_stress_test/spectral_stress_test.json)
- [`spectral_stress_test.npz`](benchmarks/spectral_stress_test/spectral_stress_test.npz)

## Findings in the recorded synthetic benchmark

The baseline autodiff Jacobian agrees with centered finite differences at
relative error `7.48e-11`. The coarse-grid three-control Jacobian has numerical
rank 3 at all three waits; the reported condition numbers are approximately
23.9, 26.0, and 18.4. Full rank does not mean all parameter combinations are
equally precise: the rate parameters show strong local correlation, reaching
about `-0.98` between `k12` and `gamma_phi` in this run. The equal-noise
two-wait information score ranks `[0.2,4.0]` best among the three tested
pairs, for both the baseline minimum-eigenvalue/log-determinant criteria and
the local trap/dephasing contrast. However, the projected trap and dephasing
directions remain very similar (cosine around 0.987 in the corresponding
design calculation), so this is complementary sampling, not complete
microscopic separation.

With moderate trap truth `k_d=0.003`, the joint C fit recovers the configured
baseline controls and trap rate with weighted SSE about `1.45e-19`; A and B
have weighted SSE about `2.64e7` and `5.06e5`, respectively, under the
benchmark's stated noise scale. B absorbs part of the trap response, including
substantial improvement at short wait, but does not represent its waiting-time
behavior consistently. For the separate per-wait B fits at this trap rate,
the short-wait weighted SSE is about `593`, increasing to `2.35e5` at `t2=4`.
These are outcomes under this synthetic model and grid, not a universal
ranking of physical mechanisms.

The reduced A model exhibits waiting-time-dependent apparent parameters for
nonzero trap truth despite its true controls remaining fixed. At `k_d=0.003`,
the short/intermediate A fits move `k12` to about `0.00722` and raise
`gamma_phi` to about `0.0242`; at `t2=4`, the fit gives `k12` about `0.01338`
and `gamma_phi` about `0.02247`, versus truth `k12=0.014` and
`gamma_phi=0.022`. For stronger traps, short/intermediate `k12` estimates hit
the lower bound while long-wait estimates remain interior. At `k_d=0.0003`,
the transfer estimate changes only modestly, from about `0.01332` at early and
intermediate wait to `0.01394` at long wait. Thus drift depends on trap
strength and wait; it is not assumed at every setting. The large residual
costs and full-rank local Jacobians in the benchmark distinguish systematic
structural mismatch from an exact null direction, while conditioning still
limits precision.

The first-order bias prediction reproduces the direction of the small-trap
fit shift in the tested case. At `t2=0.2`, the predicted transfer displacement
is `delta k12/k_d=-2.26` and the predicted dephasing displacement is
`delta gamma_phi/k_d=0.712`. At `t2=4`, the transfer displacement is much
smaller (`-0.199` per unit trap rate). For `k_d=0.012`, the reported fraction
of hidden-channel perturbation orthogonal to the A tangent rises across waits
from `0.064` to `0.243` to `0.554`. In this benchmark, the unabsorbable
spectral signature is more apparent at the long wait. The local bias formula
should not be extrapolated to large rates without checking nonlinear fits.

The reverse test fitting C to no-trap A truth returns `k_d=3.74e-12` and
baseline controls to numerical precision. The conditional trap information at
the middle wait is `2.02e11` under the benchmark noise weighting. Because
`k_d` is constrained to be nonnegative, a near-zero estimate is a boundary
result and its local covariance is not a symmetric confidence interval.
Profile likelihood or boundary-aware bootstrap is required to calibrate
evidence that trapping is unnecessary versus trapping being too weak to
observe.

The stress run shows that conclusions depend strongly on noise and window.
For `k_d=0.003`, all waits, and 5% relative noise on the full grid, the
expected tangent residual noncentrality is `1.19` for A and `0.021` after B
refits its nonnegative dephasing rate. The trap/dephasing residual directions
have cosine `0.991`; across 64 seeds B beats A's penalized score in
`40.6%` of replicates. With 0.1% noise, the same all-wait/full-grid stress
setting gives noncentrality `2967` for A and `52.8` for B, with B winning in
all 64 replicates. At 5% noise and early-plus-late waits, B wins in `22%` of
center-window replicates, `28%` of cross-peak-window replicates, and `33%` of
full-grid replicates; using all waits raises the full-grid rate to `40.6%`.
These figures illustrate both a detection regime and substantial degeneracy,
not an experimental threshold. At zero trap, B wins in `6.25%` of full-grid
replicates, an empirical false-selection frequency for this finite-seed local
score.

## Interpretation guardrails

1. **Model inconsistency:** a structured residual beyond the fitted tangent,
   or systematic waiting-time parameter drift with residuals above the noise
   assumption, is evidence that the candidate description is inadequate for
   these synthetic observations.
2. **Mechanism improves prediction:** a more complete candidate that lowers
   held-out error or noise-aware predictive loss supports adding the tested
   mechanism within the candidate set and assumed observation/noise model.
3. **Microscopic identity:** neither a reduced-model residual nor a better
   trap-model fit proves that the physical mechanism is specifically an
   irreversible dark trap. Distinct omitted physics can create similar
   spectral directions. Identity requires discriminating candidate
   mechanisms, independent observables, realistic noise modeling, and
   appropriate uncertainty analysis.

The stress data especially caution that A-versus-B discrimination can fail at
high noise even when both are locally full rank, and that the trap and
dephasing fingerprints are highly collinear. A small residual or improved
training fit cannot by itself resolve that ambiguity.

## Publication figure set

The following vector PDF figures are the assigned publication outputs; use
the adjacent PNG previews for quick inspection. These generated files are
checked into the otherwise ignored publication-figure path. The sensitivity
atlas evaluates the forward model directly on a denser 21 by 21 frequency
grid for display; it does not interpolate the smoke run's 3 by 3 spectrum or
derivatives. The dense display grid is a visualization calculation. The
reported fit, information, and stress diagnostics remain those from the
coarse grids listed above, unless a figure-specific caption explicitly gives
another evaluated grid.

- [Spectral sensitivity atlas (PDF)](figures/publication/sensitivity/figure_spectral_sensitivity_atlas.pdf) and [PNG preview](figures/publication/sensitivity/figure_spectral_sensitivity_atlas.png): baseline spectrum, parameter-resolved derivatives, and generalized sensitivity-mode perturbation.
- [Waiting-time observability (PDF)](figures/publication/sensitivity/figure_waiting_time_observability.pdf) and [PNG preview](figures/publication/sensitivity/figure_waiting_time_observability.png): information spectrum, correlations, and complementary time selection.
- [Hidden-channel model discrimination (PDF)](figures/publication/discrimination/figure_hidden_channel_model_discrimination.pdf) and [PNG preview](figures/publication/discrimination/figure_hidden_channel_model_discrimination.png): parameter drift, model residual/prediction comparison, and bias/tangent decomposition.
- [Stress design robustness (PDF)](figures/publication/discrimination/figure_stress_design_robustness.pdf) and [PNG preview](figures/publication/discrimination/figure_stress_design_robustness.png): noise/window/wait sensitivity in the adversarial Monte Carlo tangent test.

The source numerical records and the driver scripts are the provenance for
these figures. Figure rendering does not upgrade the benchmark's scientific
scope or its uncertainty calibration.

## Reproduction and validation

Run from the repository root. These commands recreate the committed smoke
records and the adversarial stress records:

```bash
python run_spectral_sensitivity_model_discrimination.py \
  --output-dir docs/benchmarks/spectral_sensitivity_model_discrimination \
  --nfreq 3 --waiting-times 0.2 1.0 4.0 \
  --trap-rates 0 0.0003 0.003 0.012 0.04 --max-nfev 35

python run_spectral_stress_test.py \
  --output-dir docs/benchmarks/spectral_stress_test \
  --nfreq 5 --waiting-times 0.2 1.0 4.0 \
  --relative-noise 0.001 0.01 0.05 --nmc 64

MPLCONFIGDIR=/tmp/mpl-sensitivity \
  python docs/figures/publication/sensitivity/generate_figures.py \
  --benchmark-dir docs/benchmarks/spectral_sensitivity_model_discrimination \
  --output-dir docs/figures/publication/sensitivity \
  --nfreq-dense 21

python docs/figures/publication/discrimination/generate_figures.py \
  --output-dir docs/figures/publication/discrimination
```

The sensitivity driver writes the directly evaluated dense plot arrays to
`docs/figures/publication/sensitivity/dense_sensitivity_plot_data.npz`; its
waiting-time diagnostics are also written to
`waiting_time_observability_plot_data.json`. The discrimination driver
regenerates the two linked main figures and supporting panels
`figure2_ABC_complex_residuals.pdf` and
`figure3_fit_and_heldout_metrics.pdf`, with PNG previews alongside each PDF.

The recorded run report states that the targeted sensitivity/structural suite
passed 17/17 tests, including the structural discrimination tests and existing
dimer spectral, observability, autodiff, and endpoint inversion checks. The
subsequent stress-test-focused suite was recorded as 19/19 passing, including
equal pixel-budget and deterministic-window checks. To rerun the focused
coverage from the repository root:

```bash
pytest -q \
  test_dimer_structural_discrimination.py \
  test_spectral_stress_test.py \
  test_dimer_spectral_observable.py \
  test_dimer_observability_integration.py \
  test_exciton_dimer_autodiff.py \
  test_dimer_endpoint_inversion.py
```

The machine-readable JSON records contain the exact benchmark parameters,
diagnostics, optimization outputs, and stated caveats for each run. Larger
frequency grids, realistic correlated noise, independent experimental
replicates, nonlinear uncertainty profiling near rate boundaries, and
held-out spectral-window acquisitions remain necessary before drawing
experimental or publication-level claims about identifiability or mechanism
identity.
