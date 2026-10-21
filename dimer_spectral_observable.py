"""Differentiable 2D spectral observable for the covariant dimer benchmark.

The endpoint controls are ``q = (J_12, k_12, gamma_phi)``.  The site-energy
gap and temperature are fixed.  Detailed-balance transfer is represented by

    k_1_to_2 = k_12 exp(-beta Delta / 2),
    k_2_to_1 = k_12 exp(+beta Delta / 2),

and collective dephasing by the Lindblad operator
``sqrt(gamma_phi) (sigma_z^(1) + sigma_z^(2))``.

The response follows the repository's column-stacked, RWA nested-commutator
conventions.  For each RP/NRP pathway it evaluates

    Tr[mu G3 J3 exp(L t2) J2 G1 J1 rho_g]

and sums the requested phase-matched sectors.  This is an exact finite-model
complex response on fixed omega1/omega3 grids; it is not yet a pulse-envelope,
orientational-average, disorder-average, or instrument-response model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
    except ModuleNotFoundError as exc:
        raise ImportError("JAX is required for the dimer spectral observable.") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, jsp


@dataclass(frozen=True)
class DimerSpectrumConfig:
    """Fixed experimental and non-inferred model settings."""

    omega1_grid: Any
    omega3_grid: Any
    mean_site_energy: float = 1.09
    detuning: float = 0.18
    beta: float = 2.5
    anharmonicity: float = -0.06
    gamma_1: float = 0.03
    gamma_2: float = 0.004
    mu_1: float = 1.0
    mu_2: float = 1.0
    mu_esa_scale: float = 1.0
    eta: float = 0.025
    sectors: tuple[str, ...] = ("RP", "NRP")


def _sector_fields(sector: str) -> tuple[str, str, str]:
    fields = {
        "RP": ("-", "+", "+"),
        "NRP": ("+", "-", "+"),
    }
    try:
        return fields[sector.upper()]
    except KeyError as exc:
        raise ValueError("Only RP and NRP sectors are supported.") from exc


def build_dimer_spectral_observable(config: DimerSpectrumConfig):
    """Return ``observable(q, t2) -> complex[omega3, omega1]``.

    The returned function is a side-effect-free JAX function and can be used
    with ``jax.jit``, ``jax.jacfwd``, ``jax.jvp``, and ``jax.vjp``.  For real
    least-squares interfaces, use :func:`pack_complex_spectrum`.
    """

    _jax, jnp, jsp = _require_jax()
    omega1_grid = jnp.asarray(config.omega1_grid, dtype=jnp.float64)
    omega3_grid = jnp.asarray(config.omega3_grid, dtype=jnp.float64)
    if omega1_grid.ndim != 1 or omega3_grid.ndim != 1:
        raise ValueError("omega1_grid and omega3_grid must be one-dimensional")
    if config.eta <= 0.0:
        raise ValueError("eta must be positive")

    dim = 4
    liouville_dim = dim * dim
    ident = jnp.eye(dim, dtype=jnp.complex128)
    ident_l = jnp.eye(liouville_dim, dtype=jnp.complex128)

    def op(i: int, j: int):
        return jnp.zeros((dim, dim), dtype=jnp.complex128).at[i, j].set(1.0)

    projectors = tuple(op(i, i) for i in range(dim))

    def left_super(operator):
        return jnp.kron(ident, operator)

    def right_super(operator):
        return jnp.kron(operator.T, ident)

    def dissipator(collapse):
        cdc = collapse.conj().T @ collapse
        return (
            jnp.kron(collapse.conj(), collapse)
            - 0.5 * jnp.kron(ident, cdc)
            - 0.5 * jnp.kron(cdc.T, ident)
        )

    mu_plus = (
        config.mu_1 * op(1, 0)
        + config.mu_2 * op(2, 0)
        + config.mu_esa_scale * config.mu_2 * op(3, 1)
        + config.mu_esa_scale * config.mu_1 * op(3, 2)
    )
    mu_minus = mu_plus.conj().T
    mu = mu_plus + mu_minus
    rho_ground = projectors[0].reshape((-1,), order="F")
    z_collective = jnp.diag(jnp.asarray([-2.0, 0.0, 0.0, 2.0]))

    interactions: dict[tuple[str, str], Any] = {}
    for side in ("L", "R"):
        for field, component in (("+", mu_minus), ("-", mu_plus)):
            interactions[(side, field)] = (
                left_super(component) if side == "L" else -right_super(component)
            )

    pathways = tuple(
        (sector.upper(), sides, _sector_fields(sector))
        for sector in config.sectors
        for sides in product(("L", "R"), repeat=3)
    )

    def liouvillian(q):
        coupling, transfer_rate, collective_dephasing = q
        omega_1 = config.mean_site_energy - 0.5 * config.detuning
        omega_2 = config.mean_site_energy + 0.5 * config.detuning
        hamiltonian = (
            omega_1 * projectors[1]
            + omega_2 * projectors[2]
            + (omega_1 + omega_2 + config.anharmonicity) * projectors[3]
            + coupling * (op(1, 2) + op(2, 1))
        )
        generator = -1j * (
            jnp.kron(ident, hamiltonian) - jnp.kron(hamiltonian.T, ident)
        )
        generator = generator + config.gamma_1 * dissipator(op(0, 1))
        generator = generator + config.gamma_1 * dissipator(op(2, 3))
        generator = generator + config.gamma_2 * dissipator(op(0, 2))
        generator = generator + config.gamma_2 * dissipator(op(1, 3))
        k_forward = transfer_rate * jnp.exp(-0.5 * config.beta * config.detuning)
        k_reverse = transfer_rate * jnp.exp(+0.5 * config.beta * config.detuning)
        generator = generator + k_forward * dissipator(op(2, 1))
        generator = generator + k_reverse * dissipator(op(1, 2))
        generator = generator + collective_dephasing * dissipator(z_collective)
        return generator

    def resolvent(generator, omega):
        return -jnp.linalg.solve(
            generator + (1j * omega - config.eta) * ident_l,
            ident_l,
        )

    def detect(state):
        density = state.reshape((dim, dim), order="F")
        return jnp.trace(mu @ density)

    def observable(q, t2):
        q = jnp.asarray(q, dtype=jnp.float64)
        generator = liouvillian(q)
        wait = jsp.linalg.expm(generator * jnp.asarray(t2, dtype=jnp.float64))
        total = jnp.zeros((omega3_grid.size, omega1_grid.size), dtype=jnp.complex128)
        for sector, sides, fields in pathways:
            j1 = interactions[(sides[0], fields[0])]
            j2 = interactions[(sides[1], fields[1])]
            j3 = interactions[(sides[2], fields[2])]
            omega1_sign = 1.0 if sector == "RP" else -1.0
            for i1 in range(omega1_grid.size):
                state = resolvent(generator, omega1_sign * omega1_grid[i1]) @ j1 @ rho_ground
                state = j3 @ wait @ j2 @ state
                for i3 in range(omega3_grid.size):
                    value = detect(resolvent(generator, -omega3_grid[i3]) @ state)
                    total = total.at[i3, i1].add(value)
        return total

    return observable


def pack_complex_spectrum(spectrum):
    """Flatten a complex spectrum into real then imaginary components."""

    _jax, jnp, _jsp = _require_jax()
    values = jnp.asarray(spectrum, dtype=jnp.complex128).reshape((-1,))
    return jnp.concatenate((jnp.real(values), jnp.imag(values)))


def build_packed_dimer_spectral_observable(config: DimerSpectrumConfig):
    """Return a real-valued vector observable convenient for inversion."""

    observable = build_dimer_spectral_observable(config)

    def packed(q, t2):
        return pack_complex_spectrum(observable(q, t2))

    return packed


__all__ = [
    "DimerSpectrumConfig",
    "build_dimer_spectral_observable",
    "build_packed_dimer_spectral_observable",
    "pack_complex_spectrum",
]
