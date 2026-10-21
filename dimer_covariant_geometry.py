"""Covariant-geometry adapter for robustness tests on the exciton dimer.

The control manifold is lambda = (omega_2 - omega_1, J_12, k_12), with the
mean site energy fixed.  The steady-state density supplies state coordinates
and the real and imaginary Liouvillian entries supply generator coordinates.
The latter retain the nonlinear detailed-balance dependence on detuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from exciton_dimer_autodiff import jax_exciton_dimer_liouvillian
from exciton_dimer_model import exciton_dimer_parameter_map

Array = np.ndarray


@dataclass(frozen=True)
class DimerGeometryBackend:
    """Compiled dimer metric, connection, and coordinate callables."""

    metric: Callable[[Array], Array]
    christoffel: Callable[[Array], Array]
    coordinates: Callable[[Array], Array]


def build_dimer_geometry_backend(
    *,
    mean_site_energy: float = 1.09,
    beta_12: float = 2.5,
    generator_scale: float = 1.0,
    control_mode: str = "spectroscopic",
    fixed_detuning: float = 0.18,
) -> DimerGeometryBackend:
    """Build JAX-autodiff geometry on a three-control dimer manifold.

    ``spectroscopic`` uses (detuning, coupling, transfer rate).
    ``coupling_transfer_dephasing`` uses (coherent coupling, detailed-balance
    transfer prefactor, collective dephasing rate). For this mode the generator
    embedding contains the Hamiltonian and the actual square-root collapse
    operators rather than only their assembled Liouvillian.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError("JAX is required for dimer covariant geometry.") from exc
    jax.config.update("jax_enable_x64", True)

    base = dict(exciton_dimer_parameter_map().reference_params)
    if control_mode not in {"spectroscopic", "coupling_transfer_dephasing"}:
        raise ValueError(
            "control_mode must be 'spectroscopic' or "
            "'coupling_transfer_dephasing'"
        )
    base["beta_12"] = float(beta_12)
    base.pop("temperature", None)
    trace_row = jnp.zeros(16, dtype=jnp.complex128).at[jnp.arange(4) * 5].set(1.0)
    steady_rhs = jnp.zeros(16, dtype=jnp.complex128).at[-1].set(1.0)

    def liouvillian_from_controls(lam):
        if control_mode == "spectroscopic":
            detuning, coupling, transfer_rate = lam
            collective_dephasing = None
        else:
            coupling, transfer_rate, collective_dephasing = lam
            detuning = fixed_detuning
        params = {
            **base,
            "omega_1": mean_site_energy - 0.5 * detuning,
            "omega_2": mean_site_energy + 0.5 * detuning,
            "J_12": coupling,
            "k_12": transfer_rate,
            "beta_12": beta_12,
        }
        if collective_dephasing is None:
            return jax_exciton_dimer_liouvillian(params)
        params = {**params, "gamma_phi_1": 0.0, "gamma_phi_2": 0.0}
        liouvillian = jax_exciton_dimer_liouvillian(params)
        ident = jnp.eye(4, dtype=jnp.complex128)
        z_sum = jnp.diag(jnp.asarray([-2.0, 0.0, 0.0, 2.0], dtype=float))
        cdc = z_sum.conj().T @ z_sum
        dissipator = (
            jnp.kron(z_sum.conj(), z_sum)
            - 0.5 * jnp.kron(ident, cdc)
            - 0.5 * jnp.kron(cdc.T, ident)
        )
        return liouvillian + collective_dephasing * dissipator

    def controlled_generator_coordinates(lam):
        coupling, transfer_rate, collective_dephasing = lam
        delta = fixed_detuning
        k_1_to_2 = transfer_rate * jnp.exp(-0.5 * beta_12 * delta)
        k_2_to_1 = transfer_rate * jnp.exp(+0.5 * beta_12 * delta)
        op_12 = jnp.zeros((4, 4), dtype=jnp.complex128).at[2, 1].set(1.0)
        op_21 = jnp.zeros((4, 4), dtype=jnp.complex128).at[1, 2].set(1.0)
        z_sum = jnp.diag(jnp.asarray([-2.0, 0.0, 0.0, 2.0], dtype=float))
        h_control = jnp.zeros((4, 4), dtype=jnp.complex128)
        h_control = h_control.at[1, 2].set(coupling)
        h_control = h_control.at[2, 1].set(coupling)
        operators = jnp.concatenate(
            (
                h_control.reshape((-1,)),
                (jnp.sqrt(k_1_to_2) * op_12).reshape((-1,)),
                (jnp.sqrt(k_2_to_1) * op_21).reshape((-1,)),
                (jnp.sqrt(collective_dephasing) * z_sum).reshape((-1,)),
            )
        )
        return jnp.concatenate((jnp.real(operators), jnp.imag(operators)))

    def coordinate_map(lam):
        liouvillian = liouvillian_from_controls(lam)
        constrained = liouvillian.at[-1, :].set(trace_row)
        rho = jnp.linalg.solve(constrained, steady_rhs)
        state_coordinates = jnp.concatenate((jnp.real(rho), jnp.imag(rho)))
        if control_mode == "spectroscopic":
            generator_coordinates = generator_scale * jnp.concatenate(
                (
                    jnp.real(liouvillian).reshape((-1,)),
                    jnp.imag(liouvillian).reshape((-1,)),
                )
            )
        else:
            generator_coordinates = generator_scale * controlled_generator_coordinates(lam)
        return jnp.concatenate((state_coordinates, generator_coordinates))

    def metric_fn(lam):
        jacobian = jax.jacfwd(coordinate_map)(lam)
        return jacobian.T @ jacobian

    def christoffel_fn(lam):
        metric = metric_fn(lam)
        dmetric = jnp.moveaxis(jax.jacfwd(metric_fn)(lam), -1, 0)
        # C[b,m,n] is the covariant first-kind Christoffel combination.
        covariant = 0.5 * (
            jnp.transpose(dmetric, (1, 0, 2))
            + jnp.transpose(dmetric, (1, 2, 0))
            - dmetric
        )
        return jnp.einsum("ab,bmn->amn", jnp.linalg.inv(metric), covariant)

    metric_jit = jax.jit(metric_fn)
    christoffel_jit = jax.jit(christoffel_fn)
    coordinates_jit = jax.jit(coordinate_map)

    def as_numpy(fn):
        return lambda lam: np.asarray(fn(jnp.asarray(lam, dtype=float)))

    # Compile once so ODE callbacks do not include tracing time.
    if control_mode == "spectroscopic":
        reference = jnp.asarray([0.18, 0.04, 0.006], dtype=float)
    else:
        reference = jnp.asarray([0.04, 0.006, 0.012], dtype=float)
    metric_jit(reference).block_until_ready()
    christoffel_jit(reference).block_until_ready()
    coordinates_jit(reference).block_until_ready()
    return DimerGeometryBackend(
        metric=as_numpy(metric_jit),
        christoffel=as_numpy(christoffel_jit),
        coordinates=as_numpy(coordinates_jit),
    )


__all__ = ["DimerGeometryBackend", "build_dimer_geometry_backend"]
