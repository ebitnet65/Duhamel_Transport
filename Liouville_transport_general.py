"""
Liouville_transport_general.py

Model-agnostic Liouville pathway transport core.

This module keeps the current transport object as the central quantity:

    |P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>.

The user supplies a Liouvillian, an initial state, and a transition dipole
operator. The code builds response pathways from nested-commutator side actions
and projects finite-control pathway states onto the lambda=0 pathway basis.
Here lambda is a dimensionless displacement from a reference/best-guess model,
not the raw physical parameter value.

The abstractions here are intentionally small. They are meant to be stable
enough to support a later Duhamel/autodiff layer without tying the core transport
code to one quantum-dynamics package.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import scipy.linalg

Array = np.ndarray
Params = Mapping[str, float]
Pathway = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ControlSpec:
    """One selected expansion/control coordinate inside a larger parameter set.

    `reference_value` is the best-guess physical parameter value. `scale` is
    the physical uncertainty or variation scale assigned to that parameter.
    The actual model parameter is

        p_mu(lambda) = reference_value + scale * lambda_mu.

    Thus lambda_mu is dimensionless and measures displacement in units of the
    chosen model-parameter uncertainty.
    """

    name: str
    reference_value: float
    scale: float = 1.0


@dataclass(frozen=True)
class ParameterMap:
    """Map a selected lambda-vector into full model parameters.

    `reference_params` is the full best-guess parameter dictionary. The selected
    controls specify which entries are allowed to move under the lambda-vector.
    Entries not listed in `controls` remain fixed at their reference values.
    """

    reference_params: Mapping[str, float]
    controls: tuple[ControlSpec, ...]

    @property
    def control_dimension(self) -> int:
        return len(self.controls)

    def params(self, lam: Array | Sequence[float] | float | None = None) -> dict[str, float]:
        """Return physical model parameters at a lambda displacement.

        `lam = 0` returns the reference/best-guess model, not a model with zero
        physical parameter values.
        """

        params = dict(self.reference_params)
        if lam is None:
            lam_array = np.zeros(self.control_dimension, dtype=float)
        else:
            lam_array = np.asarray(lam, dtype=float).reshape(-1)
        if lam_array.size != self.control_dimension:
            raise ValueError(
                f"Expected {self.control_dimension} control values, got {lam_array.size}"
            )
        for value, control in zip(lam_array, self.controls, strict=True):
            params[control.name] = control.reference_value + control.scale * float(value)
        return params

    def zero(self) -> Array:
        """Return the lambda-vector for the reference/best-guess model."""

        return np.zeros(self.control_dimension, dtype=float)

    def physical_displacements(self, lam: Array | Sequence[float] | float) -> dict[str, float]:
        """Return physical parameter shifts implied by a lambda displacement."""

        lam_array = np.asarray(lam, dtype=float).reshape(-1)
        if lam_array.size != self.control_dimension:
            raise ValueError(
                f"Expected {self.control_dimension} control values, got {lam_array.size}"
            )
        return {
            control.name: control.scale * float(value)
            for value, control in zip(lam_array, self.controls, strict=True)
        }


@dataclass(frozen=True)
class LiouvilleModel:
    """Minimal protocol container for model-agnostic Liouville transport."""

    dimension: int
    liouvillian: Callable[[Params], Array]
    initial_state: Callable[[Params], Array]
    transition_dipole: Callable[[Params], Array]
    dipole_components: Callable[[Params], tuple[Array, Array]] | None = None
    resolvent: Callable[[Array, float, float], Array] | None = None
    propagator: Callable[[Array, float], Array] | None = None

    @property
    def liouville_dimension(self) -> int:
        return self.dimension * self.dimension

    def L(self, params: Params) -> Array:
        matrix = np.asarray(self.liouvillian(params), dtype=complex)
        expected = (self.liouville_dimension, self.liouville_dimension)
        if matrix.shape != expected:
            raise ValueError(f"Liouvillian shape {matrix.shape}, expected {expected}")
        return matrix

    def rho0(self, params: Params) -> Array:
        state = np.asarray(self.initial_state(params), dtype=complex).reshape(-1)
        if state.size != self.liouville_dimension:
            raise ValueError(
                f"Initial state length {state.size}, expected {self.liouville_dimension}"
            )
        return state

    def mu(self, params: Params) -> Array:
        operator = np.asarray(self.transition_dipole(params), dtype=complex)
        expected = (self.dimension, self.dimension)
        if operator.shape != expected:
            raise ValueError(f"Transition dipole shape {operator.shape}, expected {expected}")
        return operator

    def mu_components(self, params: Params) -> tuple[Array, Array]:
        if self.dipole_components is None:
            return split_transition_dipole(self.mu(params))
        mu_plus, mu_minus = self.dipole_components(params)
        expected = (self.dimension, self.dimension)
        mu_plus = np.asarray(mu_plus, dtype=complex)
        mu_minus = np.asarray(mu_minus, dtype=complex)
        if mu_plus.shape != expected or mu_minus.shape != expected:
            raise ValueError(
                "Dipole component shapes must both be "
                f"{expected}, got {mu_plus.shape} and {mu_minus.shape}"
            )
        return mu_plus, mu_minus

    def G(self, L: Array, omega: float, eta: float) -> Array:
        if self.resolvent is not None:
            return np.asarray(self.resolvent(L, omega, eta), dtype=complex)
        ident = np.eye(L.shape[0], dtype=complex)
        return -np.linalg.inv(L + (1j * omega - eta) * ident)

    def U(self, L: Array, tau: float) -> Array:
        if self.propagator is not None:
            return np.asarray(self.propagator(L, tau), dtype=complex)
        return scipy.linalg.expm(L * tau)


@dataclass(frozen=True)
class CanonicalPathway:
    """A named pathway retained in the lambda=0 reference transport basis."""

    sector: str
    pathway: Pathway
    label: str


@dataclass(frozen=True)
class TransportResult:
    """Projected transport matrix and diagnostics at one control point."""

    T: Array
    P0: Array
    Plambda: Array
    reconstructed: Array
    residuals: Array
    singular_values: Array
    rank: int
    condition_number: float
    labels: tuple[str, ...]
    lam: Array
    stage: str


VALID_STAGES = ("after_J2", "after_U2", "after_J3", "after_G3")
SECTOR_PHASES = {
    "RP": (-1, +1, +1),
    "NRP": (+1, -1, +1),
    "DQC": (+1, +1, -1),
}


def vec(matrix: Array) -> Array:
    """Column-stack an operator."""

    return np.asarray(matrix, dtype=complex).reshape(-1, order="F")


def mat(vector: Array, dimension: int) -> Array:
    """Undo column-stacking."""

    return np.asarray(vector, dtype=complex).reshape((dimension, dimension), order="F")


def left_super(operator: Array, dimension: int | None = None) -> Array:
    """Superoperator for rho -> operator @ rho."""

    op = np.asarray(operator, dtype=complex)
    dim = op.shape[0] if dimension is None else dimension
    return np.kron(np.eye(dim, dtype=complex), op)


def right_super(operator: Array, dimension: int | None = None) -> Array:
    """Superoperator for rho -> rho @ operator."""

    op = np.asarray(operator, dtype=complex)
    dim = op.shape[0] if dimension is None else dimension
    return np.kron(op.T, np.eye(dim, dtype=complex))


def commutator_super(operator: Array, dimension: int | None = None) -> Array:
    """Superoperator for rho -> [operator, rho]."""

    return left_super(operator, dimension) - right_super(operator, dimension)


def split_transition_dipole(mu: Array) -> tuple[Array, Array]:
    """Split a transition dipole into lowering/raising components.

    The default convention uses the strictly lower triangular part as mu_plus
    and the strictly upper triangular part as mu_minus, matching the existing
    2LS code where mu_plus = |e><g| and mu_minus = |g><e|.
    """

    operator = np.asarray(mu, dtype=complex)
    mu_plus = np.tril(operator, k=-1)
    mu_minus = np.triu(operator, k=1)
    return mu_plus, mu_minus


def field_operator(
    mu: Array,
    field: str,
    dipole_components: tuple[Array, Array] | None = None,
) -> Array:
    """Return the RWA dipole component coupled to field '+' or '-'."""

    mu_plus, mu_minus = (
        split_transition_dipole(mu)
        if dipole_components is None
        else dipole_components
    )
    if field == "+":
        return mu_minus
    if field == "-":
        return mu_plus
    raise ValueError("field must be '+' or '-'")


def phase_sign_for_field(field: str) -> int:
    if field == "+":
        return +1
    if field == "-":
        return -1
    raise ValueError("field must be '+' or '-'")


def interaction_super(
    side: str,
    field: str,
    mu: Array,
    dipole_components: tuple[Array, Array] | None = None,
) -> Array:
    """Signed nested-commutator side action for one RWA interaction."""

    side = side.upper()
    component = field_operator(mu, field, dipole_components)
    if side == "L":
        return left_super(component)
    if side == "R":
        return -right_super(component)
    raise ValueError("side must be 'L' or 'R'")


def make_pathway(sides: Sequence[str], fields: Sequence[str]) -> Pathway:
    if len(sides) != len(fields):
        raise ValueError("sides and fields must have equal length")
    return tuple((side.upper(), field) for side, field in zip(sides, fields, strict=True))


def pathway_phase_signature(pathway: Pathway) -> tuple[int, ...]:
    return tuple(phase_sign_for_field(field) for _side, field in pathway)


def pathway_label(pathway: Pathway) -> str:
    return "".join(f"{side}{field}" for side, field in pathway)


def all_rwa_pathways(order: int = 3) -> list[Pathway]:
    return [
        make_pathway(sides, fields)
        for sides in product(("L", "R"), repeat=order)
        for fields in product(("+", "-"), repeat=order)
    ]


def get_sector_pathways(sector: str, order: int = 3) -> list[Pathway]:
    sector = sector.upper()
    if sector == "ALL_RWA":
        return all_rwa_pathways(order=order)
    if order != 3:
        raise ValueError("phase-matched RP/NRP/DQC sectors are currently defined for order=3")
    if sector not in SECTOR_PHASES:
        raise ValueError("sector must be RP, NRP, DQC, or ALL_RWA")
    target = SECTOR_PHASES[sector]
    return [
        pathway
        for pathway in all_rwa_pathways(order=order)
        if pathway_phase_signature(pathway) == target
    ]


def omega1_sign_for_sector(sector: str) -> int:
    sector = sector.upper()
    if sector == "RP":
        return -1
    if sector in ("NRP", "DQC"):
        return +1
    raise ValueError("omega1 sign is defined only for RP, NRP, DQC")


def validate_stage(stage: str) -> str:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of {VALID_STAGES}.")
    return stage


def pathway_state_point(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    lam: Array | Sequence[float] | float,
    omega1: float,
    omega3: float,
    pathway: Pathway,
    sector: str,
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
) -> Array:
    """Return one pathway state at a selected response stage."""

    stage = validate_stage(stage)
    sector = sector.upper()
    params = parameter_map.params(lam)
    L = model.L(params)
    rho = model.rho0(params)
    mu = model.mu(params)
    mu_components = model.mu_components(params)

    J1 = interaction_super(*pathway[0], mu=mu, dipole_components=mu_components)
    J2 = interaction_super(*pathway[1], mu=mu, dipole_components=mu_components)
    J3 = interaction_super(*pathway[2], mu=mu, dipole_components=mu_components)

    G1 = model.G(L, omega1_sign_for_sector(sector) * omega1, eta)
    G3 = model.G(L, omega3, eta)
    U2 = model.U(L, tau2)

    state = G1 @ J1 @ rho
    state = J2 @ state
    if stage == "after_J2":
        return state
    state = U2 @ state
    if stage == "after_U2":
        return state
    state = J3 @ state
    if stage == "after_J3":
        return state
    return G3 @ state


def pathway_state_spectrum(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    lam: Array | Sequence[float] | float,
    omegas: Array,
    pathway: Pathway,
    sector: str,
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
) -> Array:
    """Compute pathway states over omega1=omega3=omega."""

    return np.asarray([
        pathway_state_point(
            model,
            parameter_map,
            lam,
            omega,
            omega,
            pathway,
            sector,
            stage=stage,
            tau2=tau2,
            eta=eta,
        )
        for omega in np.asarray(omegas, dtype=float)
    ])


def canonical_pathways(
    sectors: Iterable[str] = ("RP", "NRP"),
    *,
    order: int = 3,
    label_map: Mapping[tuple[str, str], str] | None = None,
) -> list[CanonicalPathway]:
    """Build candidate canonical pathways from nested-commutator bookkeeping."""

    out = []
    for sector in sectors:
        sector_upper = sector.upper()
        for pathway in get_sector_pathways(sector_upper, order=order):
            raw_label = pathway_label(pathway)
            label = (
                label_map.get((sector_upper, raw_label), f"{sector_upper}-{raw_label}")
                if label_map is not None
                else f"{sector_upper}-{raw_label}"
            )
            out.append(CanonicalPathway(sector_upper, pathway, label))
    return out


def liouville_pathway_basis_matrix(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    lam: Array | Sequence[float] | float,
    omegas: Array,
    pathways: Sequence[CanonicalPathway],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    quadrature_weight: bool = True,
) -> Array:
    """Build a stacked pathway-state matrix with pathways as columns."""

    if len(pathways) == 0:
        raise ValueError("pathways is empty")
    omegas_array = np.asarray(omegas, dtype=float)
    sqrt_weight = (
        np.sqrt(np.gradient(omegas_array))[:, None]
        if quadrature_weight
        else np.ones((len(omegas_array), 1))
    )

    columns = []
    for item in pathways:
        states = pathway_state_spectrum(
            model,
            parameter_map,
            lam,
            omegas_array,
            item.pathway,
            item.sector,
            stage=stage,
            tau2=tau2,
            eta=eta,
        )
        columns.append((sqrt_weight * states).reshape(-1))
    return np.column_stack(columns)


def select_independent_columns(
    matrix: Array,
    labels: Sequence[str],
    *,
    rtol: float = 1e-10,
) -> tuple[list[int], list[str], Array]:
    """Greedily select linearly independent columns from a pathway matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[1] != len(labels):
        raise ValueError("Number of labels must match matrix columns")

    selected_indices: list[int] = []
    selected_labels: list[str] = []
    residuals = np.zeros(matrix.shape[1], dtype=float)
    basis = np.empty((matrix.shape[0], 0), dtype=complex)

    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        column_norm = np.linalg.norm(column)
        if column_norm < 1e-15:
            continue
        if basis.shape[1] == 0:
            selected_indices.append(index)
            selected_labels.append(labels[index])
            basis = column[:, None]
            residuals[index] = 1.0
            continue
        coefficients, *_ = np.linalg.lstsq(basis, column, rcond=None)
        residual = column - basis @ coefficients
        relative_residual = np.linalg.norm(residual) / column_norm
        residuals[index] = relative_residual
        if relative_residual > rtol:
            selected_indices.append(index)
            selected_labels.append(labels[index])
            basis = np.column_stack((basis, column))

    return selected_indices, selected_labels, residuals


def independent_canonical_pathways(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    omegas: Array,
    pathways: Sequence[CanonicalPathway],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    rtol: float = 1e-10,
) -> tuple[list[CanonicalPathway], Array]:
    """Return a rank-reduced canonical pathway list at the reference model."""

    matrix = liouville_pathway_basis_matrix(
        model,
        parameter_map,
        parameter_map.zero(),
        omegas,
        pathways,
        stage=stage,
        tau2=tau2,
        eta=eta,
    )
    labels = [item.label for item in pathways]
    indices, _selected_labels, residuals = select_independent_columns(
        matrix,
        labels,
        rtol=rtol,
    )
    return [pathways[index] for index in indices], residuals


def liouville_transport_matrix(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    lam: Array | Sequence[float] | float,
    omegas: Array,
    pathways: Sequence[CanonicalPathway],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    rcond: float = 1e-12,
) -> TransportResult:
    """Compute the pathway transport matrix at one lambda-vector value."""

    lam_array = np.asarray(lam, dtype=float).reshape(-1)
    P0 = liouville_pathway_basis_matrix(
        model,
        parameter_map,
        parameter_map.zero(),
        omegas,
        pathways,
        stage=stage,
        tau2=tau2,
        eta=eta,
    )
    Plam = liouville_pathway_basis_matrix(
        model,
        parameter_map,
        lam_array,
        omegas,
        pathways,
        stage=stage,
        tau2=tau2,
        eta=eta,
    )

    singular_values = np.linalg.svd(P0, compute_uv=False)
    relative_singular_values = singular_values / singular_values[0]
    rank = int(np.sum(relative_singular_values > rcond))
    if rank < P0.shape[1]:
        raise np.linalg.LinAlgError(
            f"Reference pathway basis is rank deficient: rank={rank}, "
            f"columns={P0.shape[1]}"
        )

    coefficients = np.linalg.pinv(P0, rcond=rcond) @ Plam
    T = coefficients.T
    reconstructed = P0 @ coefficients
    residuals = np.array([
        np.linalg.norm(Plam[:, index] - reconstructed[:, index])
        / max(np.linalg.norm(Plam[:, index]), 1e-14)
        for index in range(Plam.shape[1])
    ])
    condition_number = (
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 0
        else np.inf
    )

    return TransportResult(
        T=T,
        P0=P0,
        Plambda=Plam,
        reconstructed=reconstructed,
        residuals=residuals,
        singular_values=singular_values,
        rank=rank,
        condition_number=float(condition_number),
        labels=tuple(item.label for item in pathways),
        lam=lam_array,
        stage=stage,
    )


def finite_difference_liouvillian_derivatives(
    model: LiouvilleModel,
    parameter_map: ParameterMap,
    *,
    step: float = 1e-5,
) -> dict[tuple[int, ...], Array]:
    """Compute first and second finite-difference derivatives of L(lambda).

    This is a placeholder backend for the Duhamel/autodiff layer. It fixes the
    tensor indexing convention before a JAX backend is introduced.

    Derivatives are taken with respect to the dimensionless lambda coordinates
    at the reference/best-guess model. For a control with `scale != 1`, this is
    a scaled physical derivative.
    """

    n_controls = parameter_map.control_dimension
    zero = parameter_map.zero()
    L0 = model.L(parameter_map.params(zero))
    derivatives: dict[tuple[int, ...], Array] = {(): L0}

    for mu in range(n_controls):
        direction = np.zeros(n_controls)
        direction[mu] = step
        Lp = model.L(parameter_map.params(zero + direction))
        Lm = model.L(parameter_map.params(zero - direction))
        derivatives[(mu,)] = (Lp - Lm) / (2.0 * step)

    for mu in range(n_controls):
        for nu in range(mu, n_controls):
            emu = np.zeros(n_controls)
            enu = np.zeros(n_controls)
            emu[mu] = step
            enu[nu] = step
            if mu == nu:
                Lp = model.L(parameter_map.params(zero + emu))
                Lm = model.L(parameter_map.params(zero - emu))
                derivatives[(mu, nu)] = (Lp - 2.0 * L0 + Lm) / (step * step)
            else:
                Lpp = model.L(parameter_map.params(zero + emu + enu))
                Lpm = model.L(parameter_map.params(zero + emu - enu))
                Lmp = model.L(parameter_map.params(zero - emu + enu))
                Lmm = model.L(parameter_map.params(zero - emu - enu))
                derivatives[(mu, nu)] = (Lpp - Lpm - Lmp + Lmm) / (4.0 * step * step)

    return derivatives


__all__ = [
    "CanonicalPathway",
    "ControlSpec",
    "LiouvilleModel",
    "ParameterMap",
    "TransportResult",
    "all_rwa_pathways",
    "canonical_pathways",
    "commutator_super",
    "field_operator",
    "finite_difference_liouvillian_derivatives",
    "get_sector_pathways",
    "independent_canonical_pathways",
    "interaction_super",
    "left_super",
    "liouville_pathway_basis_matrix",
    "liouville_transport_matrix",
    "make_pathway",
    "mat",
    "omega1_sign_for_sector",
    "pathway_label",
    "pathway_phase_signature",
    "pathway_state_point",
    "pathway_state_spectrum",
    "phase_sign_for_field",
    "right_super",
    "select_independent_columns",
    "split_transition_dipole",
    "vec",
]
