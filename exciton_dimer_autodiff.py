"""
exciton_dimer_autodiff.py

JAX-compatible Liouvillian for the four-level exciton dimer.

This module mirrors `exciton_dimer_model.py` but builds Lindblad terms as
rate-weighted dissipators,

    rate * D[C_unit],

instead of collapse operators `sqrt(rate) C_unit`. The two forms are
algebraically equivalent for positive rates, and the rate-weighted form is
smoother for autodiff at a reference model with a zero transfer rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "JAX is required for the autodiff exciton dimer Liouvillian."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jnp


def jax_exciton_dimer_liouvillian(params: Mapping[str, Any]):
    """Return the JAX Liouville-space dimer generator.

    The output uses the same column-stacked convention as
    `Liouville_transport_general.liouvillian_from_hamiltonian_and_lindblad_ops`.
    """

    jnp = _require_jax()
    dim = 4
    ident = jnp.eye(dim, dtype=jnp.complex128)

    def ket(index: int):
        return ident[:, index : index + 1]

    def op(i: int, j: int):
        return ket(i) @ ket(j).conj().T

    def projector(index: int):
        return op(index, index)

    def unit_dissipator(c):
        cdc = c.conj().T @ c
        return (
            jnp.kron(c.conj(), c)
            - 0.5 * jnp.kron(ident, cdc)
            - 0.5 * jnp.kron(cdc.T, ident)
        )

    omega_1 = params["omega_1"]
    omega_2 = params["omega_2"]
    hamiltonian = (
        omega_1 * projector(1)
        + omega_2 * projector(2)
        + (omega_1 + omega_2 + params["anharmonicity"]) * projector(3)
        + params["J_12"] * (op(1, 2) + op(2, 1))
    )
    liouvillian = -1j * (
        jnp.kron(ident, hamiltonian)
        - jnp.kron(hamiltonian.T, ident)
    )

    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    gamma_phi_1 = params["gamma_phi_1"]
    gamma_phi_2 = params["gamma_phi_2"]
    delta = omega_2 - omega_1
    if "beta_12" in params:
        beta = params["beta_12"]
    else:
        beta = 1.0 / params["temperature"]
    k_12 = params["k_12"]
    k_1_to_2 = k_12 * jnp.exp(-0.5 * beta * delta)
    k_2_to_1 = k_12 * jnp.exp(+0.5 * beta * delta)

    liouvillian = liouvillian + gamma_1 * unit_dissipator(op(0, 1))
    liouvillian = liouvillian + gamma_1 * unit_dissipator(op(2, 3))
    liouvillian = liouvillian + gamma_2 * unit_dissipator(op(0, 2))
    liouvillian = liouvillian + gamma_2 * unit_dissipator(op(1, 3))
    liouvillian = liouvillian + gamma_phi_1 * unit_dissipator(projector(1) + projector(3))
    liouvillian = liouvillian + gamma_phi_2 * unit_dissipator(projector(2) + projector(3))
    liouvillian = liouvillian + k_1_to_2 * unit_dissipator(op(2, 1))
    liouvillian = liouvillian + k_2_to_1 * unit_dissipator(op(1, 2))
    return liouvillian


__all__ = ["jax_exciton_dimer_liouvillian"]
