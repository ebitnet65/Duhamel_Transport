"""
Liouville_autodiff.py

Optional autodiff backend for Liouvillian connection tensors.

The transport expansion uses dimensionless control coordinates lambda_mu around
a reference model,

    p_mu(lambda) = p_mu,ref + delta_mu lambda_mu.

This module computes derivatives of the Liouvillian with respect to those
dimensionless coordinates:

    L0
    partial_mu L
    partial_mu partial_nu L

The output dictionary intentionally matches
`finite_difference_liouvillian_derivatives` in `Liouville_transport_general`.
That keeps the finite-difference and autodiff backends interchangeable at the
Duhamel layer.

JAX is optional. Importing this module does not require JAX; calling the JAX
backend does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

import Liouville_transport_general as general

Array = np.ndarray
Params = Mapping[str, Any]


def is_jax_available() -> bool:
    """Return True when JAX can be imported in the current environment."""

    try:
        import jax  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "JAX is required for autodiff Liouvillian derivatives. Install jax "
            "in the active environment, or use "
            "`finite_difference_liouvillian_derivatives` as the fallback backend."
        ) from exc
    return jax, jnp


def params_from_lambda(
    parameter_map: general.ParameterMap,
    lam: Any,
) -> dict[str, Any]:
    """Map lambda coordinates to physical parameters without breaking autodiff.

    This is the JAX-compatible analogue of `ParameterMap.params`. It preserves
    traced scalar values instead of converting them through Python `float`.
    """

    params: dict[str, Any] = dict(parameter_map.reference_params)
    for index, control in enumerate(parameter_map.controls):
        params[control.name] = control.reference_value + control.scale * lam[index]
    return params


def _as_complex_matrix(matrix: Any, shape: tuple[int, int]) -> Array:
    out = np.asarray(matrix, dtype=complex)
    if out.shape != shape:
        raise ValueError(f"Liouvillian derivative shape {out.shape}, expected {shape}")
    return out


def jax_liouvillian_derivatives_from_lambda(
    liouvillian_from_lambda: Callable[[Any], Any],
    *,
    control_dimension: int,
    liouville_dimension: int,
    lam0: Sequence[float] | None = None,
    max_order: int = 2,
) -> dict[tuple[int, ...], Array]:
    """Return autodiff derivatives of `L(lambda)` through second order.

    `liouvillian_from_lambda` must be written with `jax.numpy` operations and
    return a `(liouville_dimension, liouville_dimension)` complex array. The
    lambda coordinates are dimensionless displacements from the reference model.

    The returned keys are:

    - `()`: `L(lambda0)`
    - `(mu,)`: `partial_mu L(lambda0)`
    - `(mu, nu)`: `partial_mu partial_nu L(lambda0)`, stored with `mu <= nu`
    """

    if max_order not in (0, 1, 2):
        raise ValueError("max_order must be 0, 1, or 2")

    jax, jnp = _require_jax()
    lam_ref = (
        jnp.zeros((control_dimension,), dtype=float)
        if lam0 is None
        else jnp.asarray(lam0, dtype=float).reshape((control_dimension,))
    )
    shape = (liouville_dimension, liouville_dimension)

    def flat_liouvillian(lam):
        matrix = jnp.asarray(liouvillian_from_lambda(lam))
        if matrix.shape != shape:
            raise ValueError(f"Liouvillian shape {matrix.shape}, expected {shape}")
        return jnp.reshape(matrix, (-1,))

    L0_flat = flat_liouvillian(lam_ref)
    derivatives: dict[tuple[int, ...], Array] = {
        (): _as_complex_matrix(np.asarray(L0_flat).reshape(shape), shape)
    }

    if max_order >= 1:
        jacobian = jax.jacfwd(flat_liouvillian)(lam_ref)
        for mu in range(control_dimension):
            derivatives[(mu,)] = _as_complex_matrix(
                np.asarray(jacobian[:, mu]).reshape(shape),
                shape,
            )

    if max_order >= 2:
        hessian = jax.jacfwd(jax.jacfwd(flat_liouvillian))(lam_ref)
        for mu in range(control_dimension):
            for nu in range(mu, control_dimension):
                derivatives[(mu, nu)] = _as_complex_matrix(
                    np.asarray(hessian[:, mu, nu]).reshape(shape),
                    shape,
                )

    return derivatives


def jax_liouvillian_derivatives(
    liouvillian_from_params: Callable[[Params], Any],
    parameter_map: general.ParameterMap,
    *,
    liouville_dimension: int,
    lam0: Sequence[float] | None = None,
    max_order: int = 2,
) -> dict[tuple[int, ...], Array]:
    """Return autodiff derivatives of `L(params(lambda))`.

    This wrapper keeps model construction close to the existing `ParameterMap`
    workflow while avoiding `ParameterMap.params`, whose conversion to Python
    floats is intentionally not autodiff-compatible.
    """

    def liouvillian_from_lambda(lam):
        return liouvillian_from_params(params_from_lambda(parameter_map, lam))

    return jax_liouvillian_derivatives_from_lambda(
        liouvillian_from_lambda,
        control_dimension=parameter_map.control_dimension,
        liouville_dimension=liouville_dimension,
        lam0=lam0,
        max_order=max_order,
    )


__all__ = [
    "is_jax_available",
    "jax_liouvillian_derivatives",
    "jax_liouvillian_derivatives_from_lambda",
    "params_from_lambda",
]
