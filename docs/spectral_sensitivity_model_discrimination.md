# Spectral sensitivity and hidden-channel model discrimination

## Scope and conventions

This benchmark reuses the repository's four-level site-product exciton dimer
and the RWA nested-commutator RP/NRP response in
`dimer_spectral_observable.py`. The state basis is ground, two one-exciton
site states, and the doubly excited state. The dipole contains ground-to-single
and single-to-double transitions; the latter use `mu_esa_scale`. RP uses
`(-,+,+)` field components and NRP uses `(+,-,+)`. Frequency-domain coherence
intervals use the established Liouvillian resolvents and the waiting interval
uses its exponential propagator. Spectra retain complex values; inference
stacks all real samples followed by all imaginary samples.

The preserved benchmark remains the original three-control point
`(J12,k12,gamma_phi)=(0.025,0.014,0.022)`, with fixed site-energy gap,
temperature, lifetime rates, dipoles, and anharmonicity. No change was made to
the original spectral observable or its reference tests.

## Dark sink and optical pathways

The enlarged physical Hilbert space has an additional terminal dark state
`|d>` and jump `sqrt(kd)|d><e1|`, where `|e1>` is the first single-exciton
site state. The dark state has zero dipole, no Hamiltonian coupling, and no
return pathway. It is a sink outside the optically accessible ground,
single-, and double-excitation manifolds. Optical pulses act only on the
active block; any population already in `|d>` is unchanged by later pulses.

For every active-block coherence or population, tracing out the terminal
state gives the exact generator contribution

`-kd/2 {P1,rho_active}`, with `P1=|e1><e1|`. The implementation applies this
term to the generator used in both frequency-domain coherence resolvents,
the population waiting propagator, and the detection resolvent. This is the
exact active-block projection of the full five-state Lindblad evolution, not a
waiting-time-only loss approximation. The full five-state trap generator is
also checked independently for trace preservation and for
`dot rho_11=-kd rho_11`, `dot rho_dd=kd rho_11`, and
`dot rho_12=-kd rho_12/2`. A separate `sqrt(gamma_d)P1` Lindblad jump is tested
and shown to preserve excited population while damping coherence.

## Sensitivity and information

The pixel Jacobian is evaluated by JAX autodiff and independently by centered
finite differences. The real inference Jacobian is the real/imaginary stack.
With independent Cartesian noise standard deviation `sigma`, the benchmark
uses `W=I/sigma` and `I=J.T W.T W J`. Singular values, numerical rank,
condition number, covariance-derived parameter correlations, and
generalized eigenmodes are recorded. The physical metric is explicitly the
constant coordinate normalization `G=diag(1/s_i^2)` with reference scales
`s=(0.05 energy, 0.015 rate, 0.025 rate)`. It makes one reference-scale
displacement in each control have unit squared length. It is a documented
normalization convention, not a waiting-time-dependent information metric or
a claim of a unique microscopic metric. Information changes with waiting time;
`G` does not. The weakest generalized mode is plotted as a spectral
perturbation alongside individual coordinate derivatives.

For experiment design, two-time information matrices are added under equal
per-pixel noise. The report ranks candidate pairs by their minimum information
eigenvalue and log determinant. These local scores are design predictions;
they do not replace the noisy inversion checks.

## Model discrimination protocol

Candidate A is the unchanged dimer. Candidate B adds local pure dephasing
`gamma_d D[P1]`. Candidate C adds the terminal trap. Synthetic truth spectra
are generated with constant baseline dimer parameters and configured trap
strengths at multiple waiting times. A fits are run independently per slice
and jointly with one shared parameter vector; A/B/C are also compared in a
joint fit. The driver records fitted parameters, local covariance intervals,
Jacobian rank and conditioning, residual arrays, joint weighted residual
cost, and predictions at additional waiting times. The cost-plus-`2p` AIC
diagnostic is only interpretable under the configured independent Gaussian
Cartesian noise model. A lower in-sample residual by itself is not evidence
that a microscopic mechanism has been identified.

Intervals are local linear Gaussian intervals. At parameter boundaries or
under strong nonlinearity they require profile or bootstrap replacement. A
poorly conditioned Jacobian can make intervals large even when optimization
converges; this is distinguished from optimizer failure by reporting
termination, gradient-based least-squares status, rank, singular values, and
residuals. The present driver is a tractable benchmark, not an exhaustive
global uncertainty analysis. It does not establish microscopic identity from
fit quality alone.

## Reproduction

From the repository root, run:

```bash
python run_spectral_sensitivity_model_discrimination.py \
  --output-dir docs/benchmarks/spectral_sensitivity_model_discrimination
```

The script accepts configurable frequency bounds and size, waiting times,
trap rates, linewidth, noise level, and optimizer evaluation limit. It writes
the compressed spectra/Jacobians to `spectral_sensitivity_benchmark.npz`,
diagnostics and fit records to `spectral_sensitivity_benchmark.json`, and the
overview figure to `spectral_sensitivity_model_discrimination.png`.

The generated JSON and NPZ are the numerical record for the exact run. The
following command regenerates the committed smoke configuration:

```bash
python run_spectral_sensitivity_model_discrimination.py \
  --output-dir docs/benchmarks/spectral_sensitivity_model_discrimination \
  --nfreq 3 --waiting-times 0.2 1.0 4.0 \
  --trap-rates 0 0.0003 0.003 0.012 0.04 --max-nfev 35
```

## Validation and interpretation

`test_dimer_structural_discrimination.py` checks baseline invariance, AD vs
finite differences, trace and trap-population accounting, distinction between
trap and dephasing, and spectral differences between candidates B and C.
Existing spectral and observability tests remain in the targeted validation
command. Generated numerical values and actual test outcomes are summarized
in the run record committed with this report. A residual component orthogonal
to the fitted tangent space is evidence of model inadequacy; a better
candidate prediction is evidence for an added dissipative mechanism under the
assumed observation/noise model. Neither fact alone identifies the microscopic
channel.

## Recorded smoke benchmark

The committed run uses a deliberately small `3 x 3` frequency grid
`[0.90, 1.09, 1.28]`, waits `[0.2, 1.0, 4.0]`, `eta=0.035`, and
`sigma=0.001`, with trap rates `[0, 0.0003, 0.003, 0.012, 0.04]`. It is a
reproducible smoke benchmark for the full pipeline, not a converged production
grid study. The complex RP, NRP, and RP+NRP spectra are separately saved.

Autodiff and centered finite differences agree to relative Jacobian error
`7.48e-11`. The three-control baseline Jacobian has rank 3 at all three waits;
its condition number ranges from `18.4` to `26.0` for this grid and noise
weight. With truth `kd=0.003`, the joint fits recover the same baseline
parameters for C (`J=0.025`, `k=0.014`, `gamma_phi=0.022`, `kd=0.003`) with
weighted SSE `1.45e-19`. The corresponding weighted SSE values are `2.64e7`
for A and `5.06e5` for B. This establishes that local pure dephasing absorbs
part, but not all, of the trap-induced response in this synthetic setup.

Independent A fits show rate dependence and a waiting-time regime change.
For `kd=0.0003`, fitted transfer is `0.01332` at `t2=0.2` and `1.0`, but
`0.01394` at `t2=4`, against constant truth `0.014`. For `kd=0.003`, the
short/intermediate fits shift to `k=0.00722` and increase `gamma_phi` to about
`0.0242`; the long-wait fit returns `k=0.01338`, `gamma_phi=0.02247`. At
stronger trapping (`kd=0.012` and `0.04`), short/intermediate fits put `k`
on its lower bound, while long-wait estimates remain interior. The residual
costs are far above the assumed noise level even where the local Jacobian is
full rank: the drift is not explained by a null parameter direction alone.
Intervals and condition numbers for every fit are in the JSON record.

For the smallest nonzero trap, the local bias prediction gives
`delta k/kd=-2.26` and `delta gamma_phi/kd=0.712` at `t2=0.2`, in close
agreement with the nonlinear fit shift at `kd=0.0003`. At `t2=4`, the predicted
transfer shift is much smaller (`-0.199 per unit kd`), consistent with the
observed near-baseline long-wait transfer estimate. At the moderate probe
`kd=0.012`, the hidden-channel orthogonal residual fractions are `0.064`,
`0.243`, and `0.554` at the three waits. The missing physics becomes less
absorbable at long wait.

For the reverse experiment, fitting C to A truth gives
`kd=3.74e-12` with a full-rank local Jacobian (condition `22.1`) and baseline
parameters recovered to numerical precision. The conditional information
`I_d|lambda` at the middle wait is `2.02e11` under the configured noise scale.
Because the rate is on a nonnegative boundary, these local numbers are not a
formal symmetric confidence interval; a one-sided profile or bootstrap is
needed for calibrated evidence that trapping is unnecessary.

Three independent complex-noise seeds recover baseline controls within about
`1.6e-6` in the joint fits for this very low configured noise. The best
two-time baseline information pair is `[0.2, 4.0]`. The local trap-versus-
dephasing contrast score also selects `[0.2, 4.0]`; the projected hidden
fingerprints have cosine `0.987`, so they remain similar and the score should
be read as a preference, not complete separation. The joint C fit predicts
held-out intermediate waits with relative errors around `3e-16`; B held-out
relative errors range from `3.0e-4` to `1.1e-3`, while A ranges from
`3.7e-3` to `6.2e-3`. A and B residual maps and all per-time held-out
predictions are recorded in the JSON. The recorded targeted suite passed
`17/17` tests across the new trap tests and existing dimer spectral,
observability, autodiff, and endpoint-inversion tests.

Independent B and C fits at each of the three `kd=0.003` truth slices are
also saved. C recovers all four parameters essentially exactly at every wait;
B reduces the `t2=0.2` weighted SSE to `593` but it grows to `2.35e5` by
`t2=4.0`, while its fitted extra dephasing rises from `0.00311` to `0.00331`.
These single-slice fits support the same point as the joint fits: B can absorb
part of the short-time response, but cannot represent the population-loss
signature consistently over waiting time.

The central qualitative result is conditional on the current coarse grid and
noise level: structural mismatch generates apparent parameter drift, and
complementary waits change how much of the hidden channel lies in the fitted
tangent space. Increasing the grid density, using realistic correlated
experimental noise, profiling boundary parameters, and validating spectral
windows against independent held-out acquisitions remain open work before
experimental claims.
