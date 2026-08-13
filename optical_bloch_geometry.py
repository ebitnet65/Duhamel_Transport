"""
Optical Bloch geometry test model.

This module is intentionally separate from the SAKE dimer transport code.  It
provides a small model where the stationary state is known in closed form, but
can also be recovered by solving the Liouvillian steady-state equation.

Controls are ordered as

    lambda = (Omega, Delta, gamma)

for the Rabi coupling, detuning, and amplitude-damping rate.  The Hamiltonian is

    H = [[0, Omega / 2],
         [Omega / 2, Delta]]

and the collapse operator is sqrt(gamma) |g><e|.  The stationary Bloch
coordinates are represented as

    r = (2 Re rho_ge, 2 Im rho_ge, rho_ee - rho_gg)

and the generator coordinates used for the physical pullback metric are

    h = (Omega, Delta, gamma).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class GeometryResult:
    """Container for local physical geometry at one control point."""

    lam: Array
    r: Array
    h: Array
    metric: Array
    two_form: Array
    christoffel: Array


def _as_control_vector(lam: Array) -> Array:
    out = np.asarray(lam, dtype=float).reshape((3,))
    if np.any(out[[0, 2]] <= 0.0):
        raise ValueError("Omega and gamma should be positive for this test model.")
    return out


def analytic_stationary_bloch(lam: Array) -> Array:
    """Closed-form stationary Bloch vector for the amplitude-damped 2LS."""

    omega, delta, gamma = _as_control_vector(lam)
    denom = 4.0 * delta**2 + 2.0 * omega**2 + gamma**2
    return np.asarray(
        [
            -4.0 * delta * omega / denom,
            2.0 * omega * gamma / denom,
            -(4.0 * delta**2 + gamma**2) / denom,
        ],
        dtype=float,
    )


def analytic_generator_coordinates(lam: Array) -> Array:
    """Generator coordinates paired with the Bloch vector."""

    return _as_control_vector(lam).copy()


def analytic_stationary_density(lam: Array) -> Array:
    """Closed-form stationary density matrix in the {|g>, |e>} basis."""

    omega, delta, gamma = _as_control_vector(lam)
    denom = 4.0 * delta**2 + 2.0 * omega**2 + gamma**2
    rho_ee = omega**2 / denom
    rho_gg = 1.0 - rho_ee
    rho_ge = (-2.0 * delta * omega + 1.0j * omega * gamma) / denom
    return np.asarray([[rho_gg, rho_ge], [np.conjugate(rho_ge), rho_ee]])


def finite_difference_jacobian(
    fn: Callable[[Array], Array],
    lam: Array,
    *,
    step: float = 1.0e-5,
) -> Array:
    """Central finite-difference Jacobian for validation without autodiff."""

    lam0 = _as_control_vector(lam)
    f0 = np.asarray(fn(lam0), dtype=float)
    jac = np.empty(f0.shape + lam0.shape, dtype=float)
    for mu in range(lam0.size):
        dlam = np.zeros_like(lam0)
        dlam[mu] = step * max(1.0, abs(lam0[mu]))
        fp = np.asarray(fn(lam0 + dlam), dtype=float)
        fm = np.asarray(fn(lam0 - dlam), dtype=float)
        jac[..., mu] = (fp - fm) / (2.0 * dlam[mu])
    return jac


def metric_from_coordinate_jacobians(dr: Array, dh: Array) -> Array:
    """Return G_mn = dr_i,m dr_i,n + dh_i,m dh_i,n."""

    return np.einsum("im,in->mn", dr, dr) + np.einsum("im,in->mn", dh, dh)


def two_form_from_coordinate_jacobians(dr: Array, dh: Array) -> Array:
    """Return pullback F_mn = dr_i,m dh_i,n - dr_i,n dh_i,m."""

    return np.einsum("im,in->mn", dr, dh) - np.einsum("in,im->mn", dr, dh)


def finite_difference_metric(lam: Array, *, step: float = 1.0e-5) -> Array:
    """Metric from closed-form r and h using finite-difference derivatives."""

    dr = finite_difference_jacobian(analytic_stationary_bloch, lam, step=step)
    dh = finite_difference_jacobian(analytic_generator_coordinates, lam, step=step)
    return metric_from_coordinate_jacobians(dr, dh)


def finite_difference_two_form(lam: Array, *, step: float = 1.0e-5) -> Array:
    """Two-form from closed-form r and h using finite-difference derivatives."""

    dr = finite_difference_jacobian(analytic_stationary_bloch, lam, step=step)
    dh = finite_difference_jacobian(analytic_generator_coordinates, lam, step=step)
    return two_form_from_coordinate_jacobians(dr, dh)


def christoffel_from_metric_derivatives(metric: Array, dmetric: Array) -> Array:
    """Assemble Gamma^a_mn from G_mn and dG_eta,m,n."""

    inv_metric = np.linalg.inv(np.asarray(metric, dtype=float))
    dmetric = np.asarray(dmetric, dtype=float)
    n_controls = inv_metric.shape[0]
    gamma = np.empty((n_controls, n_controls, n_controls), dtype=float)
    for alpha in range(n_controls):
        for mu in range(n_controls):
            for nu in range(n_controls):
                acc = 0.0
                for beta in range(n_controls):
                    acc += inv_metric[alpha, beta] * (
                        dmetric[mu, beta, nu]
                        + dmetric[nu, beta, mu]
                        - dmetric[beta, mu, nu]
                    )
                gamma[alpha, mu, nu] = 0.5 * acc
    return gamma


def finite_difference_christoffel(lam: Array, *, step: float = 1.0e-5) -> Array:
    """Christoffel symbols from the closed-form metric by metric differencing."""

    lam0 = _as_control_vector(lam)
    metric0 = finite_difference_metric(lam0, step=step)
    dmetric = np.empty((3, 3, 3), dtype=float)
    for eta in range(3):
        dlam = np.zeros_like(lam0)
        dlam[eta] = step * max(1.0, abs(lam0[eta]))
        gp = finite_difference_metric(lam0 + dlam, step=step)
        gm = finite_difference_metric(lam0 - dlam, step=step)
        dmetric[eta] = (gp - gm) / (2.0 * dlam[eta])
    return christoffel_from_metric_derivatives(metric0, dmetric)


def finite_difference_geometry(lam: Array, *, step: float = 1.0e-5) -> GeometryResult:
    """Evaluate the closed-form geometry with finite-difference derivatives."""

    lam0 = _as_control_vector(lam)
    return GeometryResult(
        lam=lam0,
        r=analytic_stationary_bloch(lam0),
        h=analytic_generator_coordinates(lam0),
        metric=finite_difference_metric(lam0, step=step),
        two_form=finite_difference_two_form(lam0, step=step),
        christoffel=finite_difference_christoffel(lam0, step=step),
    )


def is_sympy_available() -> bool:
    """Return True when SymPy can be imported."""

    try:
        import sympy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@cache
def _symbolic_geometry_functions():
    """Build lambdified closed-form G, F, and Gamma without autodiff."""

    try:
        import sympy as sp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SymPy is required for the symbolic optical Bloch geometry backend."
        ) from exc

    omega, delta, gamma = sp.symbols("omega delta gamma", positive=True, real=True)
    controls = (omega, delta, gamma)
    denom = 4 * delta**2 + 2 * omega**2 + gamma**2
    r = sp.Matrix(
        [
            -4 * delta * omega / denom,
            2 * omega * gamma / denom,
            -(4 * delta**2 + gamma**2) / denom,
        ]
    )
    h = sp.Matrix([omega, delta, gamma])
    dr = r.jacobian(controls)
    dh = h.jacobian(controls)
    metric = dr.T * dr + dh.T * dh
    two_form = dr.T * dh - dh.T * dr
    inv_metric = metric.inv()
    christoffel = sp.MutableDenseNDimArray.zeros(3, 3, 3)
    for alpha in range(3):
        for mu in range(3):
            for nu in range(3):
                expr = 0
                for beta in range(3):
                    expr += inv_metric[alpha, beta] * (
                        sp.diff(metric[beta, nu], controls[mu])
                        + sp.diff(metric[beta, mu], controls[nu])
                        - sp.diff(metric[mu, nu], controls[beta])
                    )
                christoffel[alpha, mu, nu] = expr / 2

    return (
        sp.lambdify(controls, metric, "numpy"),
        sp.lambdify(controls, two_form, "numpy"),
        sp.lambdify(controls, christoffel.tolist(), "numpy"),
    )


def symbolic_geometry(lam: Array) -> GeometryResult:
    """Evaluate exact symbolic G, F, and Gamma for the Bloch model.

    This backend uses SymPy symbolic differentiation of the closed-form
    stationary state.  It is a reference implementation, not the fastest
    production path.
    """

    lam0 = _as_control_vector(lam)
    metric_fn, two_form_fn, christoffel_fn = _symbolic_geometry_functions()
    omega, delta, gamma = lam0
    return GeometryResult(
        lam=lam0,
        r=analytic_stationary_bloch(lam0),
        h=analytic_generator_coordinates(lam0),
        metric=np.asarray(metric_fn(omega, delta, gamma), dtype=float),
        two_form=np.asarray(two_form_fn(omega, delta, gamma), dtype=float),
        christoffel=np.asarray(christoffel_fn(omega, delta, gamma), dtype=float),
    )


def is_jax_available() -> bool:
    """Return True when JAX can be imported."""

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
        raise ImportError("JAX is required for the autodiff geometry backend.") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def jax_analytic_geometry(lam: Array) -> dict[str, Array]:
    """Autodiff geometry from the closed-form stationary Bloch vector."""

    jax, jnp = _require_jax()
    lam_j = jnp.asarray(lam, dtype=float).reshape((3,))

    def r_of_lam(x):
        omega, delta, gamma = x
        denom = 4.0 * delta**2 + 2.0 * omega**2 + gamma**2
        return jnp.asarray(
            [
                -4.0 * delta * omega / denom,
                2.0 * omega * gamma / denom,
                -(4.0 * delta**2 + gamma**2) / denom,
            ],
            dtype=float,
        )

    def h_of_lam(x):
        return jnp.asarray(x, dtype=float)

    return _jax_geometry_from_coordinate_maps(jax, jnp, r_of_lam, h_of_lam, lam_j)


def jax_liouvillian(lam: Array):
    """JAX Liouvillian superoperator for the optical Bloch model."""

    _, jnp = _require_jax()
    omega, delta, gamma = jnp.asarray(lam, dtype=float).reshape((3,))
    hamiltonian = jnp.asarray(
        [[0.0, 0.5 * omega], [0.5 * omega, delta]],
        dtype=jnp.complex128,
    )
    collapse = jnp.sqrt(gamma) * jnp.asarray(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=jnp.complex128,
    )
    cd_c = collapse.conj().T @ collapse

    basis = jnp.eye(4, dtype=jnp.complex128).reshape((4, 2, 2))

    def rhs(rho):
        coherent = -1.0j * (hamiltonian @ rho - rho @ hamiltonian)
        dissipative = collapse @ rho @ collapse.conj().T
        dissipative -= 0.5 * (cd_c @ rho + rho @ cd_c)
        return coherent + dissipative

    columns = jnp.stack([rhs(basis[k]).reshape((-1,)) for k in range(4)], axis=1)
    return columns


def jax_stationary_density_from_liouvillian(lam: Array):
    """Solve L rho = 0 with Tr rho = 1 using a differentiable linear solve."""

    _, jnp = _require_jax()
    liouvillian = jax_liouvillian(lam)
    trace_row = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
    system = liouvillian.at[-1, :].set(trace_row)
    rhs = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
    rho_vec = jnp.linalg.solve(system, rhs)
    return rho_vec.reshape((2, 2))


def jax_stationary_bloch_from_liouvillian(lam: Array):
    """Bloch coordinates from the numerically solved stationary state."""

    _, jnp = _require_jax()
    rho = jax_stationary_density_from_liouvillian(lam)
    return jnp.asarray(
        [
            2.0 * jnp.real(rho[0, 1]),
            2.0 * jnp.imag(rho[0, 1]),
            jnp.real(rho[1, 1] - rho[0, 0]),
        ],
        dtype=float,
    )


def jax_liouvillian_geometry(lam: Array) -> dict[str, Array]:
    """Autodiff geometry from the Liouvillian stationary state."""

    jax, jnp = _require_jax()
    lam_j = jnp.asarray(lam, dtype=float).reshape((3,))

    def h_of_lam(x):
        return jnp.asarray(x, dtype=float)

    return _jax_geometry_from_coordinate_maps(
        jax,
        jnp,
        jax_stationary_bloch_from_liouvillian,
        h_of_lam,
        lam_j,
    )


def _jax_geometry_from_coordinate_maps(jax, jnp, r_of_lam, h_of_lam, lam):
    def metric_fn(x):
        dr = jax.jacfwd(r_of_lam)(x)
        dh = jax.jacfwd(h_of_lam)(x)
        return dr.T @ dr + dh.T @ dh

    def two_form_fn(x):
        dr = jax.jacfwd(r_of_lam)(x)
        dh = jax.jacfwd(h_of_lam)(x)
        return dr.T @ dh - dh.T @ dr

    metric = metric_fn(lam)
    dmetric = jnp.moveaxis(jax.jacfwd(metric_fn)(lam), -1, 0)
    inv_metric = jnp.linalg.inv(metric)

    def gamma_component(alpha, mu, nu):
        total = 0.0
        for beta in range(3):
            total = total + inv_metric[alpha, beta] * (
                dmetric[mu, beta, nu]
                + dmetric[nu, beta, mu]
                - dmetric[beta, mu, nu]
            )
        return 0.5 * total

    christoffel = jnp.asarray(
        [
            [[gamma_component(alpha, mu, nu) for nu in range(3)] for mu in range(3)]
            for alpha in range(3)
        ],
        dtype=float,
    )
    return {
        "lam": np.asarray(lam),
        "r": np.asarray(r_of_lam(lam)),
        "h": np.asarray(h_of_lam(lam)),
        "metric": np.asarray(metric),
        "two_form": np.asarray(two_form_fn(lam)),
        "christoffel": np.asarray(christoffel),
    }


__all__ = [
    "GeometryResult",
    "analytic_generator_coordinates",
    "analytic_stationary_bloch",
    "analytic_stationary_density",
    "christoffel_from_metric_derivatives",
    "finite_difference_christoffel",
    "finite_difference_geometry",
    "finite_difference_jacobian",
    "finite_difference_metric",
    "finite_difference_two_form",
    "is_jax_available",
    "is_sympy_available",
    "jax_analytic_geometry",
    "jax_liouvillian",
    "jax_liouvillian_geometry",
    "jax_stationary_bloch_from_liouvillian",
    "jax_stationary_density_from_liouvillian",
    "metric_from_coordinate_jacobians",
    "symbolic_geometry",
    "two_form_from_coordinate_jacobians",
]
