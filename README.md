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

### Parameter Notation

For the V-system examples we use:

- `J_bd`: coherent bright-dark exchange coupling in the Hamiltonian,
  `J_bd (|b><d| + |d><b|)`.
- `k_bd`: incoherent bright-dark transfer scale used to form detailed-balance
  Lindblad rates.
- `delta_X`: the physical displacement scale for parameter `X`.

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

mu_plus = mu_gb * B_g + mu_gd * D_g
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
