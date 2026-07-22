"""
Liouville_duhamel.py

Duhamel building blocks for Liouville-space transport expansions.

This module consumes Liouvillian derivative dictionaries of the form

    {(): L0, (mu,): partial_mu L, (mu, nu): partial_mu partial_nu L}

and constructs derivative expansions of the Green's functions and waiting-time
propagator that appear in each response pathway. The derivative dictionary can
come from either finite differences or the optional autodiff backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Sequence

import numpy as np
import scipy.linalg

Array = np.ndarray


@dataclass(frozen=True)
class OperatorDerivativeExpansion:
    """Taylor data for an operator-valued function of lambda."""

    value: Array
    first: dict[int, Array]
    second: dict[tuple[int, int], Array]
    third: dict[tuple[int, int, int], Array] = field(default_factory=dict)

    def evaluate(self, lam: Array | Sequence[float], order: int = 2) -> Array:
        """Evaluate the Taylor expansion at a dimensionless lambda vector."""

        lam_array = np.asarray(lam, dtype=float).reshape(-1)
        out = np.array(self.value, dtype=complex, copy=True)
        if order >= 1:
            for mu, matrix in self.first.items():
                out += lam_array[mu] * matrix
        if order >= 2:
            for (mu, nu), matrix in self.second.items():
                factor = 0.5 if mu == nu else 1.0
                out += factor * lam_array[mu] * lam_array[nu] * matrix
        if order >= 3:
            for (mu, nu, kappa), matrix in self.third.items():
                indices = (mu, nu, kappa)
                unique = len(set(indices))
                if unique == 1:
                    factor = 1.0 / 6.0
                elif unique == 2:
                    factor = 0.5
                else:
                    factor = 1.0
                out += factor * lam_array[mu] * lam_array[nu] * lam_array[kappa] * matrix
        return out


def _control_dimension(derivatives: dict[tuple[int, ...], Array]) -> int:
    indices = [key[0] for key in derivatives if len(key) > 0]
    if not indices:
        return 0
    return max(indices) + 1


def _second_derivative(
    derivatives: dict[tuple[int, ...], Array],
    mu: int,
    nu: int,
) -> Array:
    key = (mu, nu) if mu <= nu else (nu, mu)
    if key in derivatives:
        return derivatives[key]
    return np.zeros_like(derivatives[()])


def _third_derivative(
    derivatives: dict[tuple[int, ...], Array],
    mu: int,
    nu: int,
    kappa: int,
) -> Array:
    key = tuple(sorted((mu, nu, kappa)))
    if key in derivatives:
        return derivatives[key]
    return np.zeros_like(derivatives[()])


def _ordered_permutations(indices: tuple[int, ...]) -> list[tuple[int, ...]]:
    return list(permutations(indices))


def resolvent_derivative_expansion(
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    omega: float,
    eta: float,
    *,
    max_order: int = 2,
) -> OperatorDerivativeExpansion:
    """Build derivatives of `G(omega) = -(L + (i omega - eta) I)^-1`.

    The first and second derivatives are the algebraic Duhamel/resolvent
    insertions:

        G_mu = G L_mu G

        G_munu = G L_munu G
                + G L_mu G L_nu G
                + G L_nu G L_mu G.
    """

    if max_order not in (0, 1, 2, 3):
        raise ValueError("max_order must be 0, 1, 2, or 3")

    L0 = np.asarray(liouvillian_derivatives[()], dtype=complex)
    ident = np.eye(L0.shape[0], dtype=complex)
    G0 = -np.linalg.inv(L0 + (1j * omega - eta) * ident)
    n_controls = _control_dimension(liouvillian_derivatives)

    first: dict[int, Array] = {}
    for mu in range(n_controls):
        L_mu = liouvillian_derivatives[(mu,)]
        first[mu] = G0 @ L_mu @ G0

    second: dict[tuple[int, int], Array] = {}
    for mu in range(n_controls):
        for nu in range(mu, n_controls):
            L_mu = liouvillian_derivatives[(mu,)]
            L_nu = liouvillian_derivatives[(nu,)]
            L_munu = _second_derivative(liouvillian_derivatives, mu, nu)
            second[(mu, nu)] = (
                G0 @ L_munu @ G0
                + G0 @ L_mu @ G0 @ L_nu @ G0
                + G0 @ L_nu @ G0 @ L_mu @ G0
            )

    third: dict[tuple[int, int, int], Array] = {}
    if max_order >= 3:
        for mu in range(n_controls):
            for nu in range(mu, n_controls):
                for kappa in range(nu, n_controls):
                    accum = G0 @ _third_derivative(liouvillian_derivatives, mu, nu, kappa) @ G0
                    for pair, single in (
                        ((mu, nu), kappa),
                        ((mu, kappa), nu),
                        ((nu, kappa), mu),
                    ):
                        L_pair = _second_derivative(liouvillian_derivatives, *pair)
                        L_single = liouvillian_derivatives[(single,)]
                        accum += G0 @ L_pair @ G0 @ L_single @ G0
                        accum += G0 @ L_single @ G0 @ L_pair @ G0
                    for order in _ordered_permutations((mu, nu, kappa)):
                        accum += (
                            G0
                            @ liouvillian_derivatives[(order[0],)]
                            @ G0
                            @ liouvillian_derivatives[(order[1],)]
                            @ G0
                            @ liouvillian_derivatives[(order[2],)]
                            @ G0
                        )
                    third[(mu, nu, kappa)] = accum

    return OperatorDerivativeExpansion(value=G0, first=first, second=second, third=third)


def _gauss_legendre_interval(n_quad: int, lower: float, upper: float) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    midpoint = 0.5 * (upper + lower)
    half_width = 0.5 * (upper - lower)
    return midpoint + half_width * nodes, half_width * weights


def _single_duhamel_integral(L0: Array, insertion: Array, tau: float, n_quad: int) -> Array:
    out = np.zeros_like(L0, dtype=complex)
    nodes, weights = _gauss_legendre_interval(n_quad, 0.0, tau)
    for s, weight in zip(nodes, weights, strict=True):
        out += weight * scipy.linalg.expm(L0 * (tau - s)) @ insertion @ scipy.linalg.expm(L0 * s)
    return out


def _double_duhamel_integral(
    L0: Array,
    first_insertion: Array,
    second_insertion: Array,
    tau: float,
    n_quad: int,
) -> Array:
    """Return integral over 0 <= s2 <= s1 <= tau with ordered insertions."""

    out = np.zeros_like(L0, dtype=complex)
    outer_nodes, outer_weights = _gauss_legendre_interval(n_quad, 0.0, tau)
    for s1, outer_weight in zip(outer_nodes, outer_weights, strict=True):
        inner_nodes, inner_weights = _gauss_legendre_interval(n_quad, 0.0, float(s1))
        left = scipy.linalg.expm(L0 * (tau - s1))
        for s2, inner_weight in zip(inner_nodes, inner_weights, strict=True):
            out += (
                outer_weight
                * inner_weight
                * left
                @ first_insertion
                @ scipy.linalg.expm(L0 * (s1 - s2))
                @ second_insertion
                @ scipy.linalg.expm(L0 * s2)
            )
    return out


def _triple_duhamel_integral(
    L0: Array,
    first_insertion: Array,
    second_insertion: Array,
    third_insertion: Array,
    tau: float,
    n_quad: int,
) -> Array:
    """Return integral over 0 <= s3 <= s2 <= s1 <= tau."""

    out = np.zeros_like(L0, dtype=complex)
    outer_nodes, outer_weights = _gauss_legendre_interval(n_quad, 0.0, tau)
    for s1, outer_weight in zip(outer_nodes, outer_weights, strict=True):
        middle_nodes, middle_weights = _gauss_legendre_interval(n_quad, 0.0, float(s1))
        left = scipy.linalg.expm(L0 * (tau - s1))
        for s2, middle_weight in zip(middle_nodes, middle_weights, strict=True):
            inner_nodes, inner_weights = _gauss_legendre_interval(n_quad, 0.0, float(s2))
            middle = scipy.linalg.expm(L0 * (s1 - s2))
            for s3, inner_weight in zip(inner_nodes, inner_weights, strict=True):
                out += (
                    outer_weight
                    * middle_weight
                    * inner_weight
                    * left
                    @ first_insertion
                    @ middle
                    @ second_insertion
                    @ scipy.linalg.expm(L0 * (s2 - s3))
                    @ third_insertion
                    @ scipy.linalg.expm(L0 * s3)
                )
    return out


def propagator_duhamel_expansion(
    liouvillian_derivatives: dict[tuple[int, ...], Array],
    tau: float,
    *,
    n_quad: int = 24,
    max_order: int = 2,
) -> OperatorDerivativeExpansion:
    """Build Duhamel derivatives of `U(tau) = exp(L tau)`.

    First order:

        U_mu = int_0^tau ds exp[L0(tau-s)] L_mu exp[L0 s]

    Second order:

        U_munu = int ds exp[L0(tau-s)] L_munu exp[L0 s]
                + int_{s2 <= s1} ds1 ds2
                    exp[L0(tau-s1)] L_mu exp[L0(s1-s2)] L_nu exp[L0 s2]
                + the same ordered term with mu and nu exchanged.

    The integrals are evaluated by Gauss-Legendre quadrature. This is intended
    as a transparent reference implementation; production code can later cache
    exponentials or replace the quadrature with block-matrix Frechet methods.
    """

    if max_order not in (0, 1, 2, 3):
        raise ValueError("max_order must be 0, 1, 2, or 3")
    if n_quad < 2:
        raise ValueError("n_quad must be at least 2")

    L0 = np.asarray(liouvillian_derivatives[()], dtype=complex)
    U0 = scipy.linalg.expm(L0 * tau)
    n_controls = _control_dimension(liouvillian_derivatives)

    first: dict[int, Array] = {}
    for mu in range(n_controls):
        first[mu] = _single_duhamel_integral(
            L0,
            liouvillian_derivatives[(mu,)],
            tau,
            n_quad,
        )

    second: dict[tuple[int, int], Array] = {}
    for mu in range(n_controls):
        for nu in range(mu, n_controls):
            L_mu = liouvillian_derivatives[(mu,)]
            L_nu = liouvillian_derivatives[(nu,)]
            L_munu = _second_derivative(liouvillian_derivatives, mu, nu)
            second[(mu, nu)] = (
                _single_duhamel_integral(L0, L_munu, tau, n_quad)
                + _double_duhamel_integral(L0, L_mu, L_nu, tau, n_quad)
                + _double_duhamel_integral(L0, L_nu, L_mu, tau, n_quad)
            )

    third: dict[tuple[int, int, int], Array] = {}
    if max_order >= 3:
        for mu in range(n_controls):
            for nu in range(mu, n_controls):
                for kappa in range(nu, n_controls):
                    accum = _single_duhamel_integral(
                        L0,
                        _third_derivative(liouvillian_derivatives, mu, nu, kappa),
                        tau,
                        n_quad,
                    )
                    for pair, single in (
                        ((mu, nu), kappa),
                        ((mu, kappa), nu),
                        ((nu, kappa), mu),
                    ):
                        L_pair = _second_derivative(liouvillian_derivatives, *pair)
                        L_single = liouvillian_derivatives[(single,)]
                        accum += _double_duhamel_integral(L0, L_pair, L_single, tau, n_quad)
                        accum += _double_duhamel_integral(L0, L_single, L_pair, tau, n_quad)
                    for order in _ordered_permutations((mu, nu, kappa)):
                        accum += _triple_duhamel_integral(
                            L0,
                            liouvillian_derivatives[(order[0],)],
                            liouvillian_derivatives[(order[1],)],
                            liouvillian_derivatives[(order[2],)],
                            tau,
                            n_quad,
                        )
                    third[(mu, nu, kappa)] = accum

    return OperatorDerivativeExpansion(value=U0, first=first, second=second, third=third)


__all__ = [
    "OperatorDerivativeExpansion",
    "propagator_duhamel_expansion",
    "resolvent_derivative_expansion",
]
