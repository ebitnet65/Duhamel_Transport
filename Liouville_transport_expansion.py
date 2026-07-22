"""
Liouville_transport_expansion.py

Derivative expansion of pathway transport matrices.

This module turns pathway-state derivatives into derivatives of the transport
matrix

    |P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>.

Two derivative backends are provided:

1. finite differences of exact pathway states;
2. Duhamel insertion of Liouvillian derivative tensors through G1, U2, and G3.

Both backends return the same `TransportDerivativeExpansion` object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import Liouville_duhamel as duhamel
import Liouville_transport_general as general

Array = np.ndarray


@dataclass(frozen=True)
class TransportDerivativeExpansion:
    """Transport-matrix derivatives around the reference model."""

    T0: Array
    first: dict[int, Array]
    second: dict[tuple[int, int], Array]
    residual0: Array
    first_residuals: dict[int, Array]
    second_residuals: dict[tuple[int, int], Array]
    labels: tuple[str, ...]
    stage: str

    def evaluate(self, lam: Array | Sequence[float], order: int = 2) -> Array:
        """Evaluate the Taylor approximation for a dimensionless lambda vector."""

        lam_array = np.asarray(lam, dtype=float).reshape(-1)
        out = np.array(self.T0, dtype=complex, copy=True)
        if order >= 1:
            for mu, matrix in self.first.items():
                out += lam_array[mu] * matrix
        if order >= 2:
            for (mu, nu), matrix in self.second.items():
                factor = 0.5 if mu == nu else 1.0
                out += factor * lam_array[mu] * lam_array[nu] * matrix
        return out


@dataclass(frozen=True)
class _VectorDerivativeExpansion:
    """Taylor data for one pathway-state vector."""

    value: Array
    first: dict[int, Array]
    second: dict[tuple[int, int], Array]


def _control_dimension(derivatives: dict[tuple[int, ...], Array]) -> int:
    indices = [key[0] for key in derivatives if len(key) > 0]
    return max(indices) + 1 if indices else 0


def _required_liouvillian_keys(n_controls: int, max_order: int) -> set[tuple[int, ...]]:
    required = {()}
    if max_order >= 1:
        required.update((mu,) for mu in range(n_controls))
    if max_order >= 2:
        required.update((mu, nu) for mu in range(n_controls) for nu in range(mu, n_controls))
    return required


def _validate_liouvillian_derivatives(
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    n_controls: int,
    *,
    max_order: int,
) -> None:
    missing = _required_liouvillian_keys(n_controls, max_order).difference(
        liouvillian_derivatives
    )
    if missing:
        raise ValueError(f"Missing Liouvillian derivative keys: {sorted(missing)}")


def _constant_vector(value: Array, n_controls: int) -> _VectorDerivativeExpansion:
    vector = np.asarray(value, dtype=complex).reshape(-1)
    zeros = np.zeros_like(vector)
    return _VectorDerivativeExpansion(
        value=vector,
        first={mu: zeros.copy() for mu in range(n_controls)},
        second={
            (mu, nu): zeros.copy()
            for mu in range(n_controls)
            for nu in range(mu, n_controls)
        },
    )


def _apply_constant_operator(
    operator: Array,
    vector: _VectorDerivativeExpansion,
) -> _VectorDerivativeExpansion:
    matrix = np.asarray(operator, dtype=complex)
    return _VectorDerivativeExpansion(
        value=matrix @ vector.value,
        first={mu: matrix @ derivative for mu, derivative in vector.first.items()},
        second={key: matrix @ derivative for key, derivative in vector.second.items()},
    )


def _apply_operator_expansion(
    operator: duhamel.OperatorDerivativeExpansion,
    vector: _VectorDerivativeExpansion,
) -> _VectorDerivativeExpansion:
    """Apply an operator Taylor expansion to a vector Taylor expansion."""

    n_controls = len(vector.first)
    value = operator.value @ vector.value
    first = {
        mu: operator.first[mu] @ vector.value + operator.value @ vector.first[mu]
        for mu in range(n_controls)
    }
    second = {}
    for mu in range(n_controls):
        for nu in range(mu, n_controls):
            second[(mu, nu)] = (
                operator.second[(mu, nu)] @ vector.value
                + operator.first[mu] @ vector.first[nu]
                + operator.first[nu] @ vector.first[mu]
                + operator.value @ vector.second[(mu, nu)]
            )
    return _VectorDerivativeExpansion(value=value, first=first, second=second)


def _stacked_pathway_derivative_matrices(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omegas: Array,
    pathways: Sequence[general.CanonicalPathway],
    *,
    stage: str,
    tau2: float,
    eta: float,
    step: float,
) -> tuple[Array, dict[int, Array], dict[tuple[int, int], Array]]:
    """Finite-difference derivatives of stacked pathway matrices P(lambda)."""

    n_controls = parameter_map.control_dimension
    zero = parameter_map.zero()
    P0 = general.liouville_pathway_basis_matrix(
        model,
        parameter_map,
        zero,
        omegas,
        pathways,
        stage=stage,
        tau2=tau2,
        eta=eta,
    )
    first: dict[int, Array] = {}
    second: dict[tuple[int, int], Array] = {}

    for mu in range(n_controls):
        direction = np.zeros(n_controls)
        direction[mu] = step
        Pp = general.liouville_pathway_basis_matrix(
            model,
            parameter_map,
            zero + direction,
            omegas,
            pathways,
            stage=stage,
            tau2=tau2,
            eta=eta,
        )
        Pm = general.liouville_pathway_basis_matrix(
            model,
            parameter_map,
            zero - direction,
            omegas,
            pathways,
            stage=stage,
            tau2=tau2,
            eta=eta,
        )
        first[mu] = (Pp - Pm) / (2.0 * step)
        second[(mu, mu)] = (Pp - 2.0 * P0 + Pm) / (step * step)

    for mu in range(n_controls):
        for nu in range(mu + 1, n_controls):
            emu = np.zeros(n_controls)
            enu = np.zeros(n_controls)
            emu[mu] = step
            enu[nu] = step
            Ppp = general.liouville_pathway_basis_matrix(
                model,
                parameter_map,
                zero + emu + enu,
                omegas,
                pathways,
                stage=stage,
                tau2=tau2,
                eta=eta,
            )
            Ppm = general.liouville_pathway_basis_matrix(
                model,
                parameter_map,
                zero + emu - enu,
                omegas,
                pathways,
                stage=stage,
                tau2=tau2,
                eta=eta,
            )
            Pmp = general.liouville_pathway_basis_matrix(
                model,
                parameter_map,
                zero - emu + enu,
                omegas,
                pathways,
                stage=stage,
                tau2=tau2,
                eta=eta,
            )
            Pmm = general.liouville_pathway_basis_matrix(
                model,
                parameter_map,
                zero - emu - enu,
                omegas,
                pathways,
                stage=stage,
                tau2=tau2,
                eta=eta,
            )
            second[(mu, nu)] = (Ppp - Ppm - Pmp + Pmm) / (4.0 * step * step)

    return P0, first, second


def duhamel_pathway_derivative_matrices(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omegas: Array,
    pathways: Sequence[general.CanonicalPathway],
    *,
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    n_quad: int = 24,
    quadrature_weight: bool = True,
) -> tuple[Array, dict[int, Array], dict[tuple[int, int], Array]]:
    """Construct pathway-state derivatives from Duhamel insertions.

    This backend assumes that the transition dipole and initial state are fixed
    at the reference model. The lambda dependence is carried by the Liouvillian
    through `liouvillian_derivatives`, which may come from finite differences or
    autodiff.
    """

    stage = general.validate_stage(stage)
    n_controls = parameter_map.control_dimension
    _validate_liouvillian_derivatives(
        liouvillian_derivatives,
        n_controls,
        max_order=2,
    )
    zero = parameter_map.zero()
    params0 = parameter_map.params(zero)
    L0 = model.L(params0)
    if not np.allclose(L0, liouvillian_derivatives[()], rtol=1e-9, atol=1e-11):
        raise ValueError("liouvillian_derivatives[()] does not match the reference model L0")

    rho = model.rho0(params0)
    mu = model.mu(params0)
    mu_components = model.mu_components(params0)
    omegas_array = np.asarray(omegas, dtype=float)
    U2_expansion = (
        duhamel.propagator_duhamel_expansion(
            liouvillian_derivatives,
            tau2,
            n_quad=n_quad,
        )
        if stage != "after_J2"
        else None
    )
    sqrt_weight = (
        np.sqrt(np.gradient(omegas_array))
        if quadrature_weight
        else np.ones(len(omegas_array), dtype=float)
    )

    value_columns: list[Array] = []
    first_columns: dict[int, list[Array]] = {mu_index: [] for mu_index in range(n_controls)}
    second_columns: dict[tuple[int, int], list[Array]] = {
        (mu_index, nu_index): []
        for mu_index in range(n_controls)
        for nu_index in range(mu_index, n_controls)
    }

    for item in pathways:
        J1 = general.interaction_super(*item.pathway[0], mu=mu, dipole_components=mu_components)
        J2 = general.interaction_super(*item.pathway[1], mu=mu, dipole_components=mu_components)
        J3 = general.interaction_super(*item.pathway[2], mu=mu, dipole_components=mu_components)

        values = []
        first_values = {mu_index: [] for mu_index in range(n_controls)}
        second_values = {
            (mu_index, nu_index): []
            for mu_index in range(n_controls)
            for nu_index in range(mu_index, n_controls)
        }

        for omega in omegas_array:
            G1 = duhamel.resolvent_derivative_expansion(
                liouvillian_derivatives,
                general.omega1_sign_for_sector(item.sector) * float(omega),
                eta,
            )
            state = _constant_vector(J1 @ rho, n_controls)
            state = _apply_operator_expansion(G1, state)
            state = _apply_constant_operator(J2, state)
            if stage != "after_J2":
                state = _apply_operator_expansion(U2_expansion, state)
            if stage in ("after_J3", "after_G3"):
                state = _apply_constant_operator(J3, state)
            if stage == "after_G3":
                G3 = duhamel.resolvent_derivative_expansion(
                    liouvillian_derivatives,
                    -float(omega),
                    eta,
                )
                state = _apply_operator_expansion(G3, state)

            values.append(state.value)
            for mu_index, derivative in state.first.items():
                first_values[mu_index].append(derivative)
            for key, derivative in state.second.items():
                second_values[key].append(derivative)

        weights = sqrt_weight[:, None]
        value_columns.append((weights * np.asarray(values)).reshape(-1))
        for mu_index in range(n_controls):
            first_columns[mu_index].append(
                (weights * np.asarray(first_values[mu_index])).reshape(-1)
            )
        for key in second_columns:
            second_columns[key].append((weights * np.asarray(second_values[key])).reshape(-1))

    P0 = np.column_stack(value_columns)
    first = {key: np.column_stack(columns) for key, columns in first_columns.items()}
    second = {key: np.column_stack(columns) for key, columns in second_columns.items()}
    return P0, first, second


def duhamel_pathway_derivative_grid_matrices(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omega1_values: Array,
    omega3_values: Array,
    pathways: Sequence[general.CanonicalPathway],
    *,
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    stage: str = "after_G3",
    tau2: float = 1.0,
    eta: float = 1e-3,
    n_quad: int = 24,
    quadrature_weight: bool = True,
) -> tuple[Array, dict[int, Array], dict[tuple[int, int], Array]]:
    """Construct Duhamel pathway derivatives on a full 2D frequency grid."""

    stage = general.validate_stage(stage)
    n_controls = parameter_map.control_dimension
    _validate_liouvillian_derivatives(
        liouvillian_derivatives,
        n_controls,
        max_order=2,
    )
    zero = parameter_map.zero()
    params0 = parameter_map.params(zero)
    L0 = model.L(params0)
    if not np.allclose(L0, liouvillian_derivatives[()], rtol=1e-9, atol=1e-11):
        raise ValueError("liouvillian_derivatives[()] does not match the reference model L0")

    rho = model.rho0(params0)
    mu = model.mu(params0)
    mu_components = model.mu_components(params0)
    omega1_array = np.asarray(omega1_values, dtype=float)
    omega3_array = np.asarray(omega3_values, dtype=float)
    if quadrature_weight:
        weights = np.sqrt(np.outer(np.gradient(omega3_array), np.gradient(omega1_array)))
    else:
        weights = np.ones((len(omega3_array), len(omega1_array)), dtype=float)
    U2_expansion = (
        duhamel.propagator_duhamel_expansion(
            liouvillian_derivatives,
            tau2,
            n_quad=n_quad,
        )
        if stage != "after_J2"
        else None
    )

    value_columns: list[Array] = []
    first_columns: dict[int, list[Array]] = {mu_index: [] for mu_index in range(n_controls)}
    second_columns: dict[tuple[int, int], list[Array]] = {
        (mu_index, nu_index): []
        for mu_index in range(n_controls)
        for nu_index in range(mu_index, n_controls)
    }

    for item in pathways:
        J1 = general.interaction_super(*item.pathway[0], mu=mu, dipole_components=mu_components)
        J2 = general.interaction_super(*item.pathway[1], mu=mu, dipole_components=mu_components)
        J3 = general.interaction_super(*item.pathway[2], mu=mu, dipole_components=mu_components)

        values = []
        first_values = {mu_index: [] for mu_index in range(n_controls)}
        second_values = {
            (mu_index, nu_index): []
            for mu_index in range(n_controls)
            for nu_index in range(mu_index, n_controls)
        }

        for i3, omega3 in enumerate(omega3_array):
            for i1, omega1 in enumerate(omega1_array):
                G1 = duhamel.resolvent_derivative_expansion(
                    liouvillian_derivatives,
                    general.omega1_sign_for_sector(item.sector) * float(omega1),
                    eta,
                )
                state = _constant_vector(J1 @ rho, n_controls)
                state = _apply_operator_expansion(G1, state)
                state = _apply_constant_operator(J2, state)
                if stage != "after_J2":
                    state = _apply_operator_expansion(U2_expansion, state)
                if stage in ("after_J3", "after_G3"):
                    state = _apply_constant_operator(J3, state)
                if stage == "after_G3":
                    G3 = duhamel.resolvent_derivative_expansion(
                        liouvillian_derivatives,
                        -float(omega3),
                        eta,
                    )
                    state = _apply_operator_expansion(G3, state)

                weight = weights[i3, i1]
                values.append(weight * state.value)
                for mu_index, derivative in state.first.items():
                    first_values[mu_index].append(weight * derivative)
                for key, derivative in state.second.items():
                    second_values[key].append(weight * derivative)

        value_columns.append(np.asarray(values).reshape(-1))
        for mu_index in range(n_controls):
            first_columns[mu_index].append(np.asarray(first_values[mu_index]).reshape(-1))
        for key in second_columns:
            second_columns[key].append(np.asarray(second_values[key]).reshape(-1))

    P0 = np.column_stack(value_columns)
    first = {key: np.column_stack(columns) for key, columns in first_columns.items()}
    second = {key: np.column_stack(columns) for key, columns in second_columns.items()}
    return P0, first, second


def _project_derivative(P0: Array, derivative: Array, rcond: float) -> tuple[Array, Array]:
    """Project derivative columns onto P0 and return T-derivative orientation."""

    coefficients = np.linalg.pinv(P0, rcond=rcond) @ derivative
    reconstructed = P0 @ coefficients
    residuals = np.array([
        np.linalg.norm(derivative[:, index] - reconstructed[:, index])
        / max(np.linalg.norm(derivative[:, index]), 1e-14)
        for index in range(derivative.shape[1])
    ])
    return coefficients.T, residuals


def finite_difference_transport_expansion(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omegas: Array,
    pathways: Sequence[general.CanonicalPathway],
    *,
    liouvillian_derivatives: dict[tuple[int, ...], Array] | None = None,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    step: float = 1e-4,
    rcond: float = 1e-12,
) -> TransportDerivativeExpansion:
    """Build derivative expansion of T(lambda) around the reference model.

    `liouvillian_derivatives` is currently validated but not consumed. It is the
    interface point for the diagrammatic Duhamel implementation.
    """

    if liouvillian_derivatives is not None:
        _validate_liouvillian_derivatives(
            liouvillian_derivatives,
            parameter_map.control_dimension,
            max_order=1,
        )

    P0, first_P, second_P = _stacked_pathway_derivative_matrices(
        model,
        parameter_map,
        omegas,
        pathways,
        stage=stage,
        tau2=tau2,
        eta=eta,
        step=step,
    )
    T0, residual0 = _project_derivative(P0, P0, rcond)
    first = {}
    first_residuals = {}
    for mu, derivative in first_P.items():
        first[mu], first_residuals[mu] = _project_derivative(P0, derivative, rcond)

    second = {}
    second_residuals = {}
    for key, derivative in second_P.items():
        second[key], second_residuals[key] = _project_derivative(P0, derivative, rcond)

    return TransportDerivativeExpansion(
        T0=T0,
        first=first,
        second=second,
        residual0=residual0,
        first_residuals=first_residuals,
        second_residuals=second_residuals,
        labels=tuple(item.label for item in pathways),
        stage=stage,
    )


def duhamel_transport_expansion(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omegas: Array,
    pathways: Sequence[general.CanonicalPathway],
    *,
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1e-3,
    n_quad: int = 24,
    rcond: float = 1e-12,
) -> TransportDerivativeExpansion:
    """Build `T(lambda)` derivatives from Duhamel pathway-state insertions."""

    P0, first_P, second_P = duhamel_pathway_derivative_matrices(
        model,
        parameter_map,
        omegas,
        pathways,
        liouvillian_derivatives=liouvillian_derivatives,
        stage=stage,
        tau2=tau2,
        eta=eta,
        n_quad=n_quad,
    )
    T0, residual0 = _project_derivative(P0, P0, rcond)
    first = {}
    first_residuals = {}
    for mu, derivative in first_P.items():
        first[mu], first_residuals[mu] = _project_derivative(P0, derivative, rcond)

    second = {}
    second_residuals = {}
    for key, derivative in second_P.items():
        second[key], second_residuals[key] = _project_derivative(P0, derivative, rcond)

    return TransportDerivativeExpansion(
        T0=T0,
        first=first,
        second=second,
        residual0=residual0,
        first_residuals=first_residuals,
        second_residuals=second_residuals,
        labels=tuple(item.label for item in pathways),
        stage=stage,
    )


__all__ = [
    "TransportDerivativeExpansion",
    "duhamel_pathway_derivative_grid_matrices",
    "duhamel_pathway_derivative_matrices",
    "duhamel_transport_expansion",
    "finite_difference_transport_expansion",
]
