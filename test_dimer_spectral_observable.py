import numpy as np
import pytest
from itertools import product


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from dimer_spectral_observable import (  # noqa: E402
    DimerSpectrumConfig,
    build_dimer_spectral_observable,
    build_packed_dimer_spectral_observable,
)
import Liouville_transport_general as general  # noqa: E402


def _numpy_reference_spectrum(config, q, t2):
    """Independent reference using the established NumPy transport core."""

    coupling, transfer_rate, collective_dephasing = np.asarray(q, dtype=float)
    dim = 4

    def op(i, j):
        matrix = np.zeros((dim, dim), dtype=complex)
        matrix[i, j] = 1.0
        return matrix

    projectors = tuple(op(i, i) for i in range(dim))
    omega_1 = config.mean_site_energy - 0.5 * config.detuning
    omega_2 = config.mean_site_energy + 0.5 * config.detuning
    hamiltonian = (
        omega_1 * projectors[1]
        + omega_2 * projectors[2]
        + (omega_1 + omega_2 + config.anharmonicity) * projectors[3]
        + coupling * (op(1, 2) + op(2, 1))
    )
    k_forward = transfer_rate * np.exp(-0.5 * config.beta * config.detuning)
    k_reverse = transfer_rate * np.exp(+0.5 * config.beta * config.detuning)
    z_collective = np.diag([-2.0, 0.0, 0.0, 2.0])
    collapse_ops = (
        np.sqrt(config.gamma_1) * op(0, 1),
        np.sqrt(config.gamma_1) * op(2, 3),
        np.sqrt(config.gamma_2) * op(0, 2),
        np.sqrt(config.gamma_2) * op(1, 3),
        np.sqrt(k_forward) * op(2, 1),
        np.sqrt(k_reverse) * op(1, 2),
        np.sqrt(collective_dephasing) * z_collective,
    )
    mu_plus = (
        config.mu_1 * op(1, 0)
        + config.mu_2 * op(2, 0)
        + config.mu_esa_scale * config.mu_2 * op(3, 1)
        + config.mu_esa_scale * config.mu_1 * op(3, 2)
    )
    mu_minus = mu_plus.conj().T
    mu = mu_plus + mu_minus
    model = general.LiouvilleModel.from_hamiltonian(
        dimension=dim,
        hamiltonian=lambda _params: hamiltonian,
        collapse_ops=lambda _params: collapse_ops,
        initial_state=lambda _params: projectors[0].reshape(-1, order="F"),
        transition_dipole=lambda _params: mu,
        dipole_components=lambda _params: (mu_plus, mu_minus),
    )
    parameter_map = general.ParameterMap(reference_params={}, controls=())
    result = np.zeros(
        (len(config.omega3_grid), len(config.omega1_grid)), dtype=complex
    )
    for sector in config.sectors:
        fields = {"RP": ("-", "+", "+"), "NRP": ("+", "-", "+")}[sector]
        for sides in product(("L", "R"), repeat=3):
            pathway = general.make_pathway(sides, fields)
            for i3, omega3 in enumerate(config.omega3_grid):
                for i1, omega1 in enumerate(config.omega1_grid):
                    state = general.pathway_state_point(
                        model,
                        parameter_map,
                        np.empty(0),
                        omega1,
                        omega3,
                        pathway,
                        sector,
                        stage="after_G3",
                        tau2=t2,
                        eta=config.eta,
                    )
                    result[i3, i1] += np.trace(mu @ state.reshape((dim, dim), order="F"))
    return result


@pytest.fixture(scope="module")
def calculation():
    frequencies = np.linspace(0.88, 1.30, 4)
    config = DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    complex_observable = build_dimer_spectral_observable(config)
    packed_observable = build_packed_dimer_spectral_observable(config)
    q = jnp.asarray([0.04, 0.006, 0.012], dtype=jnp.float64)
    return config, complex_observable, packed_observable, q


def test_observable_is_deterministic_complex128_grid(calculation):
    config, observable, _, q = calculation
    first = observable(q, 0.8)
    second = observable(q, 0.8)
    assert first.shape == (len(config.omega3_grid), len(config.omega1_grid))
    assert first.dtype == jnp.complex128
    assert np.all(np.isfinite(np.asarray(first)))
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(np.asarray(first)) > 1e-8


def test_endpoint_controls_and_waiting_time_change_spectrum(calculation):
    _, observable, _, q = calculation
    base = np.asarray(observable(q, 0.8))
    shifted = np.asarray(observable(q + jnp.asarray([0.012, 0.003, 0.006]), 0.8))
    later = np.asarray(observable(q, 1.6))
    assert np.linalg.norm(shifted - base) / np.linalg.norm(base) > 1e-3
    assert np.linalg.norm(later - base) / np.linalg.norm(base) > 1e-3


def test_real_packed_observable_supports_jacobian_jvp_and_vjp(calculation):
    _, _, observable, q = calculation
    t2 = 0.8
    tangent = jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float64)
    jacobian = jax.jacfwd(lambda x: observable(x, t2))(q)
    value, jvp = jax.jvp(lambda x: observable(x, t2), (q,), (tangent,))
    _, pullback = jax.vjp(lambda x: observable(x, t2), q)
    cotangent = jnp.linspace(-0.2, 0.4, value.size, dtype=jnp.float64)
    vjp = pullback(cotangent)[0]

    assert value.dtype == jnp.float64
    assert jacobian.dtype == jnp.float64
    assert jacobian.shape == (value.size, 3)
    assert np.all(np.isfinite(np.asarray(jacobian)))
    assert np.linalg.matrix_rank(np.asarray(jacobian), tol=1e-9) == 3
    np.testing.assert_allclose(jvp, jacobian @ tangent, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(vjp, jacobian.T @ cotangent, rtol=2e-10, atol=2e-10)


def test_jacobian_matches_centered_finite_difference(calculation):
    _, _, observable, q = calculation
    t2 = 0.8
    jacobian = np.asarray(jax.jacfwd(lambda x: observable(x, t2))(q))
    steps = np.asarray([2e-6, 2e-7, 2e-7])
    columns = []
    for index, step in enumerate(steps):
        direction = np.zeros(3)
        direction[index] = step
        plus = np.asarray(observable(q + direction, t2))
        minus = np.asarray(observable(q - direction, t2))
        columns.append((plus - minus) / (2.0 * step))
    finite_difference = np.column_stack(columns)
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-5, atol=2e-6)


def test_spectrum_matches_independent_numpy_transport_core():
    frequencies = np.asarray([0.94, 1.22])
    config = DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    q = np.asarray([0.04, 0.006, 0.012])
    t2 = 0.8
    observed = np.asarray(build_dimer_spectral_observable(config)(q, t2))
    expected = _numpy_reference_spectrum(config, q, t2)
    np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=2e-11)
