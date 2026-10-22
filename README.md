# SAKE-DT: Duhamel Transport

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21793438.svg)](https://doi.org/10.5281/zenodo.21793438)

**SAKE-DT** is a Spectral Autodiff Kernel Expansion framework for Duhamel
transport of Liouville-space nonlinear-response pathways.

This checkout is the isolated `codex/bloch-geometry-autodiff` development line
for covariant SAKE. The publication branches `codex/sake-autodiff-dimer` and
`codex/jcp3-finite-difference` are preserved unchanged.

## Covariant Geometry Test Branch

The optical Bloch amplitude-damping model provides the analytic reference for
the covariant geometry layer. Its closed-form stationary state makes it
possible to validate symbolic, finite-difference, closed-form autodiff, and
Liouvillian-autodiff calculations against one another.

The validated workflow is:

\[
\lambda
\rightarrow
\mathcal L(\lambda)
\rightarrow
\rho_{\mathrm{ss}}(\lambda)
\rightarrow
r(\lambda),h(\lambda)
\rightarrow
G_{\mu\nu}(\lambda),F_{\mu\nu}(\lambda),\Gamma^\alpha_{\mu\nu}(\lambda).
\]

### Implemented geometry and transport

The branch currently includes:

- `optical_bloch_geometry.py`: closed-form, symbolic, finite-difference, and
  JAX/Liouvillian geometry backends.
- `test_optical_bloch_geometry.py`: validation tests comparing the closed-form
  and Liouvillian-autodiff routes.
- `demo_optical_bloch_geometry.py`: command-line diagnostic driver.
- `Liouville_covariant_transport.py`: model-independent Levi-Civita transport
  of tangent vectors and frames along a supplied path.
- `Liouville_geodesics.py`: fixed-endpoint discrete path-energy minimization,
  control bounds, and metric determinant/condition diagnostics.
- `dimer_covariant_geometry.py`: JAX geometry adapters used to test the same
  transport layer on the four-level exciton dimer.
- `test_liouville_covariant_transport.py` and `test_liouville_geodesics.py`:
  invariant, reversibility, Euclidean-reference, and refinement checks.
- `test_dimer_covariant_transport.py`, `test_dimer_geodesic_transport.py`, and
  `test_dimer_three_control_geodesic.py`: new-model robustness and
  straight-versus-geodesic comparisons.
- `docs/optical_bloch_geometry_test_branch.md`: equations, backend details,
  transport results, and benchmark notes.
- `docs/covariant_sake_paper_plan.md`: stand-alone manuscript positioning,
  required contributions, validation hierarchy, figure plan, and decision
  gates relative to the under-review JCPIV_SAKE paper.

Parallel transport integrates

\[
\frac{dV^\alpha}{ds}
=-\Gamma^\alpha_{\mu\nu}(\lambda(s))
  \dot\lambda^\mu V^\nu.
\]

The implementation preserves metric Gram matrices and recovers the initial
frame under reversed-path transport.

### Dimer validation log

The first curved dimer manifold used controls
\((\omega_2-\omega_1,J_{12},k_{12})\) with stationary-density coordinates and
real/imaginary Liouvillian generator coordinates. The optimized physical path
had lower energy than straight interpolation and produced a distinct endpoint
frame, while both transports preserved metric inner products.

The next three-control benchmark used

\[
\lambda=(J,k,\gamma_\phi)
\]

at fixed detuning and temperature, with detailed-balance transfer operators

\[
C_{1\to2}=\sqrt{k e^{-\beta\Delta/2}}|01\rangle\langle10|,
\qquad
C_{2\to1}=\sqrt{k e^{+\beta\Delta/2}}|10\rangle\langle01|,
\]

and collective dephasing

\[
C_\phi=\sqrt{\gamma_\phi}
(\sigma_z^{(1)}+\sigma_z^{(2)}).
\]

For the tested endpoints, the nine-vertex geodesic lowered path energy by
about 5.59 percent. The rate coordinates curved away from straight
interpolation while \(J\) remained linear. This square-root operator embedding
is intrinsically flat, so its endpoint parallel transport is path independent
even though the minimum-energy path is curved in rate coordinates.

### Control-dimensional scaling

`benchmark_covariant_scaling.py` measures geometry compilation/evaluation,
geodesic discovery, and frame transport for 2--16 controls at fixed path
resolution. The empirical sweep found

\[
T_{\mathrm{geodesic}}\sim n^{1.89},
\qquad
N_{\mathrm{objective\ calls}}\sim n^{1.04}.
\]

The finite-difference path optimizer is the first scaling bottleneck. Warm
metric/Christoffel evaluation and frame transport remained small through 16
controls. Detailed results are in `docs/covariant_scaling_analysis.md` and
`covariant_scaling_benchmark/`.

The next performance upgrade is a fully JAX discrete-energy objective with an
autodiff gradient supplied directly to L-BFGS-B, followed by batched metric
evaluation over path nodes. A later benchmark must vary control dimension,
path resolution, and spectral-pixel count independently.

### Covariant endpoint-discovery milestone

The experimental truth observable is an endpoint property and does not depend
on the path:

\[
Y_{\mathrm{truth}}=\mathcal O(q_{\mathrm{true}}).
\]

The first complete covariant benchmark now keeps endpoint inference and route
discovery separate:

1. Discover the endpoint from the fixed 2D spectral observation,

   \[
   q^\star=\arg\min_q
   \frac12\|W[\mathcal O(q)-Y_{\mathrm{truth}}]\|_2^2+R(q).
   \]

2. Validate spectral recovery and, for synthetic data, direct recovery of
   \(q_{\mathrm{true}}\).
3. With \(q^\star\) fixed, solve the boundary-value problem
   \(\gamma(0)=p\), \(\gamma(1)=q^\star\) by minimizing physical path energy.
4. Parallel-transport a control-manifold tangent frame along
   \(\gamma_{p q^\star}\).

The implemented synthetic recovery test omits geodesic-distance regularization,
so the reference model cannot bias endpoint discovery. It uses deterministic
bounded multistart L-BFGS-B with exact JAX gradients. The recovered endpoint is
then frozen before the geodesic boundary-value problem is solved.

For the archived three-control dimer test,

\[
q_{\mathrm{true}}=(0.025,0.014,0.022),
\qquad
p=(0.070,0.003,0.004),
\]

the inversion reached a relative spectral residual of \(1.26\times10^{-9}\).
The nine-vertex geodesic reduced physical path energy by 4.86 percent relative
to coordinate-space straight interpolation while reaching the recovered
endpoint exactly. Tangent-frame metric compatibility was preserved to better
than \(10^{-10}\).

This is the central distinction from the earlier JCPIV_SAKE calculation. That
work established local spectral/autodiff expansion and endpoint recovery but
did not construct a physical control-manifold metric, solve the geodesic
boundary-value problem, or parallel-transport a frame on that manifold. The
present milestone adds those geometric operations downstream of a strictly
path-independent spectral inversion.

The present parallel transport is deliberately limited to tangent frames of
the physical control manifold. Parallel transport of the actual Liouville
pathway frame remains the next major milestone; it must compare the ordered
product of local pathway-transport maps along the geodesic with the
straight-line construction used in the earlier SAKE workflow.

The checked-in numerical record, reproduction commands, and limitations are
in `docs/benchmarks/covariant_endpoint_transport/README.md`.

### Fourier-window robustness and publication figures

The first controlled observation-model mismatch test applies symmetric
exponential apodization in the Fourier acquisition layer while fitting with the
ideal Liouvillian spectrum. It shows that \(k_{12}\) is especially sensitive to
the Fourier window, with systematic accompanying shifts in \(J_{12}\) and
\(\gamma_\phi\). The ideal model can remain full rank and return a converged
endpoint after a tangent-orthogonal residual has already signaled model
inadequacy.

The scientific interpretation, exact production parameters, and limitations
are documented in `docs/fourier_window_publication_test.md`. Regenerate all
final figures and compact audit data with:

```bash
python generate_fourier_window_publication_artifacts.py \
  --input-dir fourier_artifact_robustness_benchmark \
  --output-dir publication_artifacts/fourier_window
```

Endpoint robustness will be characterized using the spectral residual
Jacobian, multi-start recovery, synthetic-noise ensembles, bootstrap/profile
analyses, and singular values. The spectral identifiability matrix

\[
I=J_{\mathcal O}^{T}W^{T}WJ_{\mathcal O}
\]

must remain conceptually separate from the physical Liouvillian metric
\(G_{\mu\nu}\). Samples from the endpoint uncertainty distribution can then be
mapped to a bundle of geodesics from \(p\), propagating inference uncertainty
into route and transported-frame uncertainty.

### Waiting-time robustness benchmark

Endpoint recovery must also be tested against the population waiting time
\(t_2\). The truth endpoint remains one fixed, discoverable model
\(q_{\mathrm{true}}\); changing \(t_2\) changes the observation protocol, not
the endpoint or the physical control vector:

\[
Y_{\mathrm{truth}}(t_2)=\mathcal O(q_{\mathrm{true}};t_2).
\]

The archived noiseless benchmark completes the first test below. The timing
misspecification and shared noisy multi-time inversions remain planned:

1. **Correctly specified independent recovery.** Generate and fit each spectrum
   at the same \(t_2\). The recovered endpoints \(q^\star(t_2)\) should agree
   with the common \(q_{\mathrm{true}}\) within numerical and noise uncertainty.
2. **Waiting-time misspecification.** Generate data at
   \(t_2^{\mathrm{truth}}\) and fit it using displaced waiting times. This
   measures the endpoint bias
   \(q^\star(t_2^{\mathrm{fit}})-q_{\mathrm{true}}\) caused by timing or model
   error.
3. **Joint multi-time recovery.** Fit one shared endpoint to a stack of spectra,

   \[
   q^\star=\arg\min_q
   \frac12\sum_j
   \|W_j[\mathcal O(q;t_{2,j})-Y_{\mathrm{truth}}(t_{2,j})]\|_2^2.
   \]

The report will include endpoint drift, normalized spectral residual,
spectral-Jacobian singular values, covariance estimates, optimizer success,
and runtime versus \(t_2\). A joint fit should improve identifiability when
different waiting times expose complementary population-transfer and
dephasing information. Very short and very long waiting times are expected to
be less informative than the intermediate dynamical regime and must be
reported rather than silently excluded.

Only after the shared endpoint has been recovered will the physical geodesic
from \(p\) to \(q^\star\) be computed. Thus the robustness test cannot make the
truth observable path dependent or allow the route solver to bias endpoint
discovery.

### Planned model-adequacy and discrimination benchmark

A third robustness axis is structural model mismatch. Endpoint inversion will
be repeated with a nested family of candidate models rather than assuming that
the truth generator is known. The benchmark matrix will include:

1. **Under-specified candidate:** the synthetic truth contains a physical term
   absent from the inversion model.
2. **Correctly specified candidate:** candidate and truth use the same terms.
3. **Over-specified candidate:** the candidate contains one or more unnecessary
   controls whose truth values are zero or otherwise inactive.

The initial spectral inversion is expected to diagnose these cases through a
combination of residual structure and parameter identifiability. An
under-specified model should leave reproducible, frequency- and
waiting-time-dependent residuals. In an over-specified model, unnecessary
controls should appear as weak spectral-Jacobian directions, broad profile or
posterior uncertainty, and unstable individual parameter estimates even when
the predicted spectrum is stable. An extra parameter need not numerically
return to zero when it is correlated with retained controls, so parameter value
alone is not a sufficient discrimination criterion.

Candidate models will be compared using:

- training residuals and held-out spectral residuals at waiting times not used
  in the fit;
- singular values and effective rank of the spectral Jacobian;
- profile likelihoods or covariance/posterior widths for added controls;
- noise-aware likelihood-ratio tests for nested models when regularity
  conditions hold;
- AIC/BIC or predictive cross-validation as secondary complexity penalties;
- multi-start stability of the recovered endpoint and predicted spectrum.

The preferred model is the smallest candidate that explains both fitted and
held-out 2D spectra without structured residuals. Joint multi-\(t_2\) data are
especially valuable because a spurious parameter degeneracy at one waiting
time may be broken at another.

Geometric comparisons across candidate models require care because their
control manifolds can have different dimensions. Raw coordinate paths will not
be compared directly. Instead, candidate endpoints and geodesics will be
mapped into common observable, stationary-state, or Liouvillian embeddings;
the comparison will use predicted spectra, path energies within each model,
transported subspaces, principal angles, and uncertainty-aware endpoint
images. This keeps model discrimination in the initial inversion while using
covariant transport as a downstream diagnostic rather than as a source of
truth labels.

Run the focused covariant validation:

```bash
pytest -q \
  test_optical_bloch_geometry.py \
  test_liouville_covariant_transport.py \
  test_liouville_geodesics.py \
  test_dimer_covariant_transport.py \
  test_dimer_geodesic_transport.py \
  test_dimer_three_control_geodesic.py \
  test_benchmark_covariant_scaling.py
```

Generate a diagnostic output directory:

```bash
python demo_optical_bloch_geometry.py \
  --omega 0.35 \
  --delta 0.12 \
  --gamma 0.08 \
  --output-dir optical_bloch_geometry_demo
```

The finite-difference-only two-level-system archive for the JCP3 manuscript is
maintained separately on `codex/jcp3-finite-difference`.

The central calculation projects finite-parameter pathway states onto a
canonical lambda-zero Liouville pathway basis:

```text
|P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>
```

The resulting transport matrices `T_ab(lambda)` quantify how phase-matched
pathway components deform under a displaced Liouvillian.

## SAKE Release

The branch release note is:

- `docs/sake_autodiff_dimer_release.md`

Suggested release tag:

```text
v0.1-sake-autodiff-dimer
```

Release DOI: [10.5281/zenodo.21793438](https://doi.org/10.5281/zenodo.21793438).

## Beta Notebooks

The recommended collaborator entry points are:

- `notebooks/01_model_and_signal_demo.ipynb`: constructs the four-level exciton
  dimer, defines the lambda controls, retains RP/NRP pathway bases, and plots
  reference/target spectra.
- `notebooks/02_duhamel_transport_beta_analysis.ipynb`: computes autodiff
  Liouvillian connection tensors, builds a small order-3 Duhamel transport
  demo, inspects `T(lambda)`, and displays the production/benchmark plots when
  those artifacts are present locally.
- `notebooks/03_custom_model_workflow.ipynb`: tutorial template for a
  user-defined Liouvillian; it defines a V-system model in notebook cells,
  selects control parameters, computes Duhamel transport matrices, and compares
  target and transported final signals.
- `notebooks/04_iterative_sake_target_recovery.ipynb`: reproduces the
  three-target trust-region recovery benchmark and its transport diagnostics.
- `docs/benchmarks/iterative_inverse/README.md`: documents the archived
  benchmark data, dense spectral-evolution products, and the sector-selective
  pathway result: GSB--SE mixing in RP versus GSB--ESA mixing in NRP.
- `docs/benchmarks/iterative_timing/README.md`: separates the cost of one direct
  response from initial-chart construction and every accepted recovery step.
- `docs/autodiff_nodal_structure.md`: graph/nodal view of how autodiff sees
  `lambda -> L(lambda)`, how Duhamel consumes Liouvillian derivatives, and how
  the transport matrix produces final signals.
- `docs/sake_autodiff_dimer_release.md`: branch-specific release note for the
  SAKE/autodiff dimer development line, including production and benchmark
  commands.

## Main Files

- `phase_matched_response.py`: phase-matched RP/NRP two-level response model.
- `Liouville_transport.py`: pathway-state construction, basis projection, and
  transport-matrix diagnostics.
- `Liouville_transport_general.py`: model-agnostic transport core where users
  supply a Liouvillian, initial state, transition dipole operator, and selected
  control variables.
- `Liouville_transport_expansion.py`: derivative expansion of the pathway
  transport matrix `T(lambda)` around the reference model.
- `Liouville_autodiff.py`: optional JAX backend for Liouvillian connection
  tensors `L`, `partial_mu L`, and `partial_mu partial_nu L` with the same
  dictionary convention as the finite-difference backend.
- `Liouville_duhamel.py`: Duhamel building blocks that convert Liouvillian
  derivative tensors into resolvent and waiting-time propagator derivative
  expansions.
- `exciton_dimer_model.py`: four-level site-basis exciton dimer model with
  `|00>`, `|10>`, `|01>`, and `|11>` states, including optional ESA dipoles.
- `exciton_dimer_autodiff.py`: JAX-compatible dimer Liouvillian using
  rate-weighted Lindblad dissipators for smooth Liouvillian derivatives.
- `benchmark_dimer_transport_backends.py`: timing and accuracy comparison for
  finite-difference pathway transport versus autodiff/Duhamel transport.
- `plot_dimer_backend_scaling.py`: combines benchmark CSV files and plots
  timing/error scaling for finite-difference and autodiff/Duhamel backends.
- `fit_v_system_transport_signal.py`: proof-of-concept inverse fit that
  generates a 2D V-system target signal and recovers the control displacement
  with the transported-signal expansion.
- `diagnose_v_system_spectra.py`: V-system spectral sanity check comparing no
  bright-dark mixing, coherent mixing, and coherent plus incoherent transfer.
- `diagnose_exciton_dimer_spectra.py`: site-basis exciton dimer check with
  `mu = mu_site (|10><00| + |01><00| + h.c.)`, comparing both signs of `J`.
- `demo_dimer_duhamel_transport.py`: proof-of-concept transport/Duhamel demo
  comparing exact displaced dimer pathway signals with transported
  reconstructions from the reference basis.
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

## V-System Transport Fit

The script `fit_v_system_transport_signal.py` tests the current proof-of-concept
transport expansion as an inverse problem. It builds the V-system with two
selected controls,

```python
ControlSpec("J_bd", reference_value=0.0, scale=0.02)
ControlSpec("k_bd", reference_value=0.006, scale=0.003)
```

then draws a random dimensionless target lambda-vector, computes the exact 2D
third-order target signal, and minimizes the mismatch to a signal reconstructed
from the lambda-zero pathway basis and the second-order transport expansion.

Run a quick smoke example:

```bash
python fit_v_system_transport_signal.py --n-omega 10 --output-dir transport_fit_output_smoke
```

Run the default validation calculation on a 40 by 40 frequency grid:

```bash
python fit_v_system_transport_signal.py
```

The output directory contains:

- `fit_summary.txt`: target lambda, fitted lambda, lambda error, and relative
  signal errors.
- `fit_v_system_transport_signal.npz`: numerical arrays for later analysis.
- `reference_signal_*.png` and `target_signal_*.png`: direct full-commutator
  2D response panels. These are the spectra to inspect visually.
- `reference_pathway_sum_signal_*.png`, `fit_pathway_sum_signal_*.png`, and
  `fit_pathway_sum_residual_*.png`: the current pathway-transport proof of
  concept used for the lambda fit.

This script uses finite-difference pathway derivatives. It is meant to preserve
the analysis workflow while the next revision replaces that derivative backend
with autodiff and the diagrammatic Duhamel expansion.

## V-System Spectral Check

The default V-system treats `|D>` as optically dark:

```text
mu = mu_gb (|B><g| + |g><B|),    mu_gd = 0
```

Use the spectral diagnostic to compare no mixing, coherent `J_bd` mixing, and
coherent mixing plus detailed-balance incoherent transfer:

```bash
python diagnose_v_system_spectra.py
```

The script writes side-by-side real and magnitude panels plus difference plots
relative to the no-mixing case. It also reports window amplitudes around the
nominal `BB`, `BD`, `DB`, and `DD` coordinates. With `mu_gd = 0`, the coherent
dark-state contribution appears first in the mixing-induced satellite contrast
rather than as a direct dark transition.

## Exciton Dimer Spectral Check

For a site-basis exciton dimer, use the four-level model in
`exciton_dimer_model.py`:

```text
|0> = |00>,  |1> = |10>,  |2> = |01>,  |3> = |11>

mu_plus = mu_1 |10><00| + mu_2 |01><00|
        + mu_2 mu_esa_scale |11><10|
        + mu_1 mu_esa_scale |11><01|
```

Run:

```bash
python diagnose_exciton_dimer_spectra.py
```

The diagnostic compares `J = 0`, both signs of `J` with ESA off/on, and both
signs with ESA plus incoherent transfer. Set `mu_esa_scale = 0` to suppress
excited-state absorption paths, or leave the default `mu_esa_scale = 1` to
include them.

With the current Hamiltonian convention

```text
H_site = omega_1 |10><10| + omega_2 |01><01|
       + (omega_1 + omega_2 + Delta) |11><11|
       + J_12 (|10><01| + |01><10|)
```

positive `J_12` brightens the upper exciton in the equal-site, parallel-dipole
limit. If the desired convention is that positive `J` lowers the bright
symmetric exciton, the Hamiltonian coupling should be written with the opposite
sign, `-J`.

## Dimer Transport Demo

The proof-of-concept transport demo uses the four-level dimer with ESA on and
compares an exact displaced pathway signal against a reconstruction from the
lambda-zero pathway basis:

```bash
python demo_dimer_duhamel_transport.py
```

By default, the demo now generates contour plots on a `0.8-1.4 eV` window with
`n_omega = 40` points along each frequency axis and a finite Green's-function
broadening `eta = 3e-3`. For quick iteration, use a smaller grid:

```bash
python demo_dimer_duhamel_transport.py --n-omega 24 --output-dir dimer_duhamel_transport_demo_n24
```

The derivative backend is selected with `--backend`. The default is
finite-difference pathway transport; `--backend duhamel_autodiff` uses JAX
Liouvillian derivatives and Duhamel insertions to construct the same pathway
transport expansion:

- exact reference and target model spectra;
- exact target pathway signal;
- exact projected transport reconstruction;
- first-, second-, and optional third-order transport reconstructions;
- fitted correction values, reported as input-vs-fit lambda and physical
  parameter displacements;
- residual maps;
- transport-matrix heat maps;
- an error curve along the target lambda direction.

First- and second-derivative finite-difference steps are intentionally separate.
The first derivative uses `--fd-step` and defaults to `5e-5`. The curvature
terms use `--fd-second-step` and default to `1e-2`; this avoids the severe
`1 / h^2` noise amplification that appears when second derivatives are formed
with an overly small step. Pure second derivatives use a five-point stencil,
while mixed derivatives use the centered four-corner stencil.

Spectrum plots are written for the real, imaginary, and absolute-value
components. The `sector_RP_*` and `sector_NRP_*` reference/target plots are
direct phase-matched spectra summed over all canonical pathways in that sector.
The matching `sector_*_distorted_estimate_*` plots are the retained-basis
transport reconstructions, so they are the approximation being tested rather
than another exact sector calculation.

Wave-vector conservation is enforced in the transport step by building
independent sector blocks. For example, with `--sectors RP NRP`, the RP and NRP
pathway bases are selected, transported, and expanded separately; only the final
sector signals are summed for plotting. This prevents the transport projection
from mixing phase-matching sectors.

All plotted/saved spectra are normalized so that

```text
int d omega_1 d omega_3 |S(omega_3, omega_1)|^2 = 1.
```

For pathway reconstruction plots, the exact target pathway-signal norm is used
for the exact target, reconstructed target, and residual so their amplitudes can
be compared directly.

The default controls are `lambda = (J_12, k_12)` around the reference model:

```text
J_12(lambda_J) = 0.0   + 0.045 lambda_J
k_12(lambda_k) = 0.006 + 0.012 lambda_k
```

The default target is a moderate local displacement,
`lambda = (-0.25, 0.25)`, chosen to test the local expansion without leaving
the reference pathway subspace too aggressively.

The reference model is explicitly the `J_12 = 0` model. The exact model spectra
are the "model experimental" spectra; the transported pathway signals are the
objects used to validate the Duhamel/transport expansion.

A full plotted order-3 Duhamel/autodiff production analysis can be generated
with:

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

The `demo_summary.txt` file includes an `input-vs-fit corrections` table. It
compares the injected control displacements with the best-fit displacements
found by minimizing the transported pathway-signal residual. For the current
two-control demo, `J_12` and `k_12` are somewhat correlated in this metric, so
the fitted corrections should be read as a recoverability diagnostic rather
than as a claim that the inverse problem is uniquely conditioned.

The report separates these ideas explicitly:

- `Injected target corrections used to generate target spectra`: the exact
  lambda values, physical deltas, and target parameters used to build the model
  spectra.
- `Input target versus inverse-fit diagnostic`: the optional recovered
  correction values inferred by minimizing the residual.
- `Transport reconstruction diagnostics at injected target lambda`: the
  reconstruction errors evaluated at the known injected target.
- `Numerical setup`: start time, stop time, elapsed time, frequency window,
  finite `eta`, finite-difference steps, and phase-matching sector settings.
- `Spectral normalization`: raw integrated powers used to normalize the saved
  spectra.

The demo also writes both the Taylor-expanded transport matrix and the exact
projected transport matrix at the target lambda. Use
`transport_matrix_order*_abs.png` and
`transport_matrix_exact_projected_abs.png` together when checking whether the
transport expansion is physically reasonable. The summary reports
`||T_order|| / ||T_exact_projected||` and
`max|T_order| / max|T_exact_projected|`; values much larger than one usually
mean the derivative stencil is numerically unstable rather than that the
Liouvillian deformation is truly enormous.

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

`Liouville_transport_general.py` separates the reference/best-guess model
parameters from the selected lambda-vector controls. The lambda coordinates are
dimensionless displacements from that reference model, usually measured in
chosen uncertainty or variation scales. For example, a model may require `w0`,
`gamma_phi`, `gamma1`, and `theta`, while the expansion is performed only with
respect to uncertain controls such as `theta` or `(theta, gamma_phi)`.

The user-facing pieces are:

- `LiouvilleModel`: supplies `L(params)`, `rho0(params)`, and the transition
  dipole `mu(params)`. For QuTiP-style model construction, use
  `LiouvilleModel.from_hamiltonian(...)` with `H(params)` and `c_ops(params)`.
- `ParameterMap`: maps a lambda displacement into the full parameter
  dictionary.
- `ControlSpec`: names each selected control coordinate, its reference value,
  and its physical scale.
- `canonical_pathways`: builds nested-commutator RWA pathways.
- `liouville_transport_matrix`: computes `T(lambda)` by projecting finite
  pathway states onto the lambda-zero pathway basis.

The current derivative helper,
`finite_difference_liouvillian_derivatives`, fixes the indexing convention for
first and second derivatives with respect to selected controls. It is intended
as the reference backend before adding autodiff. These derivatives are with
respect to dimensionless lambda coordinates at the best-guess model, not
necessarily raw physical parameters.

`Liouville_autodiff.py` now provides the first optional autodiff backend for
the same object. The user supplies a JAX-compatible Liouvillian function,
usually written as `L(params)` with `jax.numpy` operations, and the module
returns a derivative dictionary with the same keys:

```python
derivatives = jax_liouvillian_derivatives(
    L_from_params,
    parameter_map,
    liouville_dimension=model.liouville_dimension,
)

L0 = derivatives[()]
dL_dlambda_J = derivatives[(0,)]
d2L_dlambda_J_dlambda_k = derivatives[(0, 1)]
```

The backend differentiates with respect to dimensionless lambda coordinates,
so `ControlSpec.scale` is included automatically through
`p(lambda) = p_ref + scale * lambda`. This means `derivatives[(mu,)]` is the
connection term `partial L / partial lambda_mu`, not the raw derivative with
respect to the physical parameter. JAX is optional: importing the repository
does not require it, but calling the autodiff backend does.

`Liouville_duhamel.py` is the next layer down in the expansion. Given the
Liouvillian derivative dictionary, it constructs operator-valued Taylor data for
the pieces that appear inside every response pathway:

```python
G1 = resolvent_derivative_expansion(derivatives, omega=omega1, eta=eta)
U2 = propagator_duhamel_expansion(derivatives, tau=tau2)

G1_at_lambda = G1.evaluate(lambda_vec, order=2)
U2_at_lambda = U2.evaluate(lambda_vec, order=2)
```

The resolvent derivatives are algebraic insertions such as
`G_mu = G L_mu G`. The propagator derivatives use explicit Duhamel integrals,
with Gauss-Legendre quadrature as a transparent reference implementation. This
is the tested core needed for the pathway transport expansion.

`Liouville_transport_expansion.py` projects pathway-state derivatives onto the
reference pathway basis to build

```text
T_ab(lambda) = delta_ab
             + lambda_mu T_ab,mu
             + 1/2 lambda_mu lambda_nu T_ab,munu
             + ...
```

Two pathway-derivative backends are available. The finite-difference backend
computes derivatives of exact pathway states directly and remains useful as the
numerical validation reference. The Duhamel backend consumes Liouvillian
derivative tensors and computes pathway-state derivatives diagrammatically from
`partial_mu L` and `partial_mu partial_nu L`.

The first Duhamel pathway backend is now available as
`duhamel_transport_expansion(...)`:

```python
derivatives = finite_difference_liouvillian_derivatives(model, parameter_map)
# or:
# derivatives = jax_liouvillian_derivatives(L_from_params, parameter_map, ...)

expansion = duhamel_transport_expansion(
    model,
    parameter_map,
    omegas,
    pathways,
    liouvillian_derivatives=derivatives,
    stage="after_U2",
)
```

This constructs pathway-state derivatives by inserting `L_mu` and `L_munu`
through `G1`, `U2`, and, for later stages, `G3`, then projects those derivatives
onto the reference pathway basis to obtain `T_mu` and `T_munu`. The current
implementation assumes that the initial state and transition dipole are fixed at
the reference model; lambda-dependence enters through the Liouvillian. This
matches the clean dimer controls `(J_12, k_12)`.

For the full 2D signal grids used in the V-system fit and dimer demo,
`fit_v_system_transport_signal.py` also exposes
`duhamel_transport_expansion_2d(...)`. It has the same return type as
`finite_difference_transport_expansion_2d(...)`, but computes the pathway-grid
derivatives from Duhamel insertions instead of finite-differencing complete
pathway grids. The production dimer script still defaults to the
finite-difference backend until the Duhamel grid backend has been benchmarked on
larger grids.

To compare the finite-difference and autodiff/Duhamel backends on the clean
dimer:

```bash
python benchmark_dimer_transport_backends.py --n-omega 12 16 24 --n-quad 16
```

For a production-scale Duhamel-only timing run:

```bash
python benchmark_dimer_transport_backends.py \
  --n-omega 40 80 120 \
  --n-quad 16 \
  --backends duhamel_autodiff
```

The benchmark reports backend CPU time, exact projected transport error, first-
and second-order reconstruction errors, and transport-matrix errors. The exact
projected transport calculation is kept as a common benchmark cost and reported
separately from the derivative backend time.

Third-order Duhamel terms are available by adding `--max-order 3` to the
benchmark command. The current third-order implementation is model agnostic and
uses the same derivative dictionary convention, including optional
`(mu, nu, eta)` Liouvillian tensors from autodiff. The resolvent third-order
terms are algebraic and cheap. The waiting-time propagator currently uses an
explicit triple Duhamel quadrature, so order 3 should be treated as a reference
implementation for small grids until that propagator derivative is optimized
with cached exponentials or block/Frechet methods.

To make timing/error scaling plots from existing benchmark summaries:

```bash
python plot_dimer_backend_scaling.py \
  dimer_backend_benchmarks/production_compare_n12_n16_n24/benchmark_summary.csv \
  dimer_backend_benchmarks/duhamel_autodiff_production_n40_n80_n120/benchmark_summary.csv \
  dimer_backend_benchmarks/20260722_163259_duhamel_autodiff_order3_n120_q4/benchmark_summary.csv \
  --output-dir dimer_backend_benchmarks/scaling_plots_autodiff_vs_fd
```

### Parameter Notation

For the V-system examples we use:

- `J_bd`: coherent bright-dark exchange coupling in the Hamiltonian,
  `J_bd (|b><d| + |d><b|)`.
- `k_bd`: incoherent bright-dark transfer scale used to form detailed-balance
  Lindblad rates.
- `delta_X`: the physical displacement scale for parameter `X`.
- `mu_gd = 0`: the default dark state has no direct transition dipole from the
  ground state; dark spectral weight must be induced dynamically.

In code, `ControlSpec.scale` stores this `delta_X`. For example:

```python
ControlSpec("J_bd", reference_value=0.0, scale=0.02)
ControlSpec("k_bd", reference_value=0.006, scale=0.003)
```

means

```text
J_bd(lambda_J) = 0.0   + delta_J * lambda_J,  delta_J = 0.02
k_bd(lambda_k) = 0.006 + delta_k * lambda_k,  delta_k = 0.003
```

We avoid the symbol `sigma_X` for these scales because sigma can be confused
with a variance or standard deviation in statistical contexts.

The test file `test_v_system_general.py` gives a concrete 3-level V-system
example with states `|g>`, `|b>`, and `|d>`. The model defines:

- a Hamiltonian with bright/dark energies and optional bright-dark coupling;
- radiative decay from `|b>` and `|d>` to `|g>`;
- independent pure dephasing on `|b>` and `|d>`;
- incoherent `|b> <-> |d>` transfer through Lindblad operators `|d><b|`
  and `|b><d|`, with rates constrained by detailed balance using either
  `temperature` or an explicit inverse temperature `beta_bd`;
- an explicit transition dipole split into `mu_plus` and `mu_minus`.

That explicit dipole-component split is important for systems where the ground
state is not ordered above the excited states in the matrix basis.

### Model Construction Pattern

The general interface is designed to support a QuTiP-like workflow without
making QuTiP a required dependency. The recommended pattern is:

1. Define Hilbert-space basis states.
2. Build projectors and transition operators transparently.
3. Assemble the Hamiltonian from those operators.
4. Assemble Lindblad collapse operators `C_k` and convert them to
   dissipators of the form
   `C_k rho C_k^dag - 1/2 {C_k^dag C_k, rho}`.
5. Define the transition dipole `mu`, and when needed explicitly define
   `mu_plus` and `mu_minus`.
6. Wrap these functions in `LiouvilleModel`.

The V-system test follows this pattern. In abbreviated form:

```python
g = ket(0, 3)
b = ket(1, 3)
d = ket(2, 3)

Pg = g @ g.conj().T
Pb = b @ b.conj().T
Pd = d @ d.conj().T

B_d = b @ d.conj().T
D_b = d @ b.conj().T
B_g = b @ g.conj().T
D_g = d @ g.conj().T

H = omega_b * Pb + omega_d * Pd + J_bd * (B_d + D_b)

mu_plus = mu_gb * B_g
mu_minus = mu_plus.conj().T
mu = mu_plus + mu_minus

c_ops = [
    sqrt(gamma_b) * (g @ b.conj().T),
    sqrt(gamma_d) * (g @ d.conj().T),
    sqrt(gamma_phi_b) * Pb,
    sqrt(gamma_phi_d) * Pd,
    sqrt(k_b_to_d) * D_b,
    sqrt(k_d_to_b) * B_d,
]
```

This is intentionally close to how one would write the model in QuTiP with
`basis`, `Qobj`, and collapse operators. A future QuTiP adapter should convert
those `Qobj` definitions at the boundary, while the transport core continues to
operate on its own matrix protocol.

The optional bridge module `Liouville_qutip_adapter.py` provides
`qutip_model_from_hamiltonian(...)`. It accepts callables that return either
QuTiP `Qobj` instances or ordinary array-like matrices, converts them at the
boundary, and returns a `LiouvilleModel`. This keeps QuTiP interoperability
available without making QuTiP a required dependency.

## Notes

Generated plots and diagnostic output directories are ignored by git. Recreate
them with the run scripts above.

## Dark-State Truth and Best Transported Dimer

The paper comparison in
[`publication_artifacts/dark_state_transport/README.md`](publication_artifacts/dark_state_transport/README.md)
uses a terminal optically dark sink as the truth and the original three-control
dimer as the sink-free Duhamel-transport candidate. The dedicated production
driver writes publication PDFs, PNG previews, and auditable complex spectra:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_transport_figures.py \
  --output-dir publication_artifacts/dark_state_transport \
  --nfreq 21 --projection-nfreq 5 --waiting-times 0.2 1.0 4.0 \
  --n-quad 2 --order 3 --trap-rate 0.003
```

Higher trap-rate cases are saved separately under
`publication_artifacts/dark_state_transport/trap_sweep/`. Once those
directories contain completed runs, summarize the full sweep with:

```bash
MPLCONFIGDIR=/tmp/mpl-dark-state python publication_artifacts/dark_state_transport/generate_dark_state_trap_sweep.py
```

For the completed initial sweep and its quantitative caveats, see the
`Trap-rate stress sweep` section of the dark-state technical notes. The checked-in
overview figure is
`publication_artifacts/dark_state_transport/trap_sweep/figure_dark_state_trap_rate_sweep.pdf`.
