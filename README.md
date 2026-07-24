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
- `docs/autodiff_nodal_structure.md`: graph/nodal view of how autodiff sees
  `lambda -> L(lambda)`, how Duhamel consumes Liouvillian derivatives, and how
  the transport matrix produces final signals.

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
