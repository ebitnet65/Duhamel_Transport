"""
Liouville_transport_expansion.py

Derivative expansion of pathway transport matrices.

This module turns pathway-state derivatives into derivatives of the transport
matrix

    |P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>.

The current implementation computes pathway-state derivatives by finite
differences of the exact pathway construction. It accepts Liouvillian derivative
tensors to fix the interface for the forthcoming diagrammatic/Duhamel
implementation, where those tensors will drive the expansion of G1, U2, and G3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

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
        required = {()}
        required.update((mu,) for mu in range(parameter_map.control_dimension))
        missing = required.difference(liouvillian_derivatives)
        if missing:
            raise ValueError(f"Missing Liouvillian derivative keys: {sorted(missing)}")

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


__all__ = [
    "TransportDerivativeExpansion",
    "finite_difference_transport_expansion",
]
