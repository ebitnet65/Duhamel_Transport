import numpy as np
import pytest

import optical_bloch_geometry as obg


def test_analytic_stationary_density_matches_bloch_coordinates():
    lam = np.asarray([0.35, 0.12, 0.08])

    rho = obg.analytic_stationary_density(lam)
    bloch = obg.analytic_stationary_bloch(lam)

    np.testing.assert_allclose(np.trace(rho), 1.0)
    np.testing.assert_allclose(rho, rho.conj().T)
    np.testing.assert_allclose(
        bloch,
        np.asarray(
            [
                2.0 * np.real(rho[0, 1]),
                2.0 * np.imag(rho[0, 1]),
                np.real(rho[1, 1] - rho[0, 0]),
            ]
        ),
    )


def test_finite_difference_metric_is_symmetric_and_positive():
    lam = np.asarray([0.35, 0.12, 0.08])

    geometry = obg.finite_difference_geometry(lam)

    np.testing.assert_allclose(geometry.metric, geometry.metric.T, atol=1e-10)
    np.testing.assert_allclose(geometry.two_form, -geometry.two_form.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(geometry.metric) > 0.0)
    np.testing.assert_allclose(
        geometry.christoffel,
        np.swapaxes(geometry.christoffel, 1, 2),
        atol=5e-5,
    )


def test_jax_liouvillian_stationary_state_matches_closed_form_when_available():
    if not obg.is_jax_available():
        pytest.skip("JAX is not installed in this environment.")

    lam = np.asarray([0.35, 0.12, 0.08])

    actual = np.asarray(obg.jax_stationary_density_from_liouvillian(lam))
    expected = obg.analytic_stationary_density(lam)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jax_liouvillian_geometry_matches_closed_form_geometry_when_available():
    if not obg.is_jax_available():
        pytest.skip("JAX is not installed in this environment.")

    lam = np.asarray([0.35, 0.12, 0.08])

    analytic = obg.jax_analytic_geometry(lam)
    liouvillian = obg.jax_liouvillian_geometry(lam)

    for key in ("r", "h", "metric", "two_form", "christoffel"):
        np.testing.assert_allclose(
            liouvillian[key],
            analytic[key],
            rtol=1e-10,
            atol=1e-10,
            err_msg=key,
        )


def test_finite_difference_geometry_matches_jax_geometry_when_available():
    if not obg.is_jax_available():
        pytest.skip("JAX is not installed in this environment.")

    lam = np.asarray([0.35, 0.12, 0.08])

    finite_difference = obg.finite_difference_geometry(lam, step=3e-5)
    analytic = obg.jax_analytic_geometry(lam)

    np.testing.assert_allclose(
        finite_difference.metric,
        analytic["metric"],
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        finite_difference.two_form,
        analytic["two_form"],
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        finite_difference.christoffel,
        analytic["christoffel"],
        rtol=2e-4,
        atol=2e-4,
    )


def test_symbolic_geometry_matches_jax_geometry_when_available():
    if not obg.is_sympy_available():
        pytest.skip("SymPy is not installed in this environment.")
    if not obg.is_jax_available():
        pytest.skip("JAX is not installed in this environment.")

    lam = np.asarray([0.35, 0.12, 0.08])

    symbolic = obg.symbolic_geometry(lam)
    analytic = obg.jax_analytic_geometry(lam)

    np.testing.assert_allclose(symbolic.metric, analytic["metric"], atol=1e-12)
    np.testing.assert_allclose(symbolic.two_form, analytic["two_form"], atol=1e-12)
    np.testing.assert_allclose(
        symbolic.christoffel,
        analytic["christoffel"],
        rtol=1e-10,
        atol=1e-10,
    )
