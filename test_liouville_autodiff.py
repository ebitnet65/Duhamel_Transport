import numpy as np
import pytest

import Liouville_autodiff as autodiff
import Liouville_transport_general as general


def _parameter_map():
    return general.ParameterMap(
        reference_params={"J": 0.1, "k": 0.2, "fixed": 3.0},
        controls=(
            general.ControlSpec("J", reference_value=0.1, scale=0.05),
            general.ControlSpec("k", reference_value=0.2, scale=0.01),
        ),
    )


def test_params_from_lambda_matches_dimensionless_control_convention():
    pmap = _parameter_map()
    lam = np.array([2.0, -3.0])

    params = autodiff.params_from_lambda(pmap, lam)

    np.testing.assert_allclose(params["J"], 0.2)
    np.testing.assert_allclose(params["k"], 0.17)
    assert params["fixed"] == 3.0


def test_jax_backend_reports_clear_missing_dependency_when_unavailable():
    if autodiff.is_jax_available():
        pytest.skip("JAX is installed; missing-dependency behavior is not active.")

    with pytest.raises(ImportError, match="JAX is required"):
        autodiff.jax_liouvillian_derivatives_from_lambda(
            lambda lam: np.eye(2) * lam[0],
            control_dimension=1,
            liouville_dimension=2,
        )


def test_jax_liouvillian_derivatives_for_quadratic_model_when_available():
    if not autodiff.is_jax_available():
        pytest.skip("JAX is not installed in this environment.")

    import jax.numpy as jnp

    pmap = _parameter_map()
    A0 = jnp.asarray([[1.0 + 0.0j, 0.0], [0.0, 2.0 + 0.0j]])
    AJ = jnp.asarray([[0.0 + 0.0j, 1.0], [2.0, 0.0 + 0.0j]])
    Ak = jnp.asarray([[3.0 + 0.0j, 0.0], [0.0, 4.0 + 0.0j]])
    AJK = jnp.asarray([[0.0 + 0.0j, 5.0], [6.0, 0.0 + 0.0j]])

    def L_from_params(params):
        delta_J = params["J"] - 0.1
        delta_k = params["k"] - 0.2
        return A0 + delta_J * AJ + delta_k * Ak + delta_J * delta_k * AJK

    derivatives = autodiff.jax_liouvillian_derivatives(
        L_from_params,
        pmap,
        liouville_dimension=2,
    )

    assert set(derivatives) == {(), (0,), (1,), (0, 0), (0, 1), (1, 1)}
    np.testing.assert_allclose(derivatives[()], np.asarray(A0))
    np.testing.assert_allclose(derivatives[(0,)], 0.05 * np.asarray(AJ))
    np.testing.assert_allclose(derivatives[(1,)], 0.01 * np.asarray(Ak))
    np.testing.assert_allclose(derivatives[(0, 1)], 0.05 * 0.01 * np.asarray(AJK))
    np.testing.assert_allclose(derivatives[(0, 0)], np.zeros((2, 2)))
    np.testing.assert_allclose(derivatives[(1, 1)], np.zeros((2, 2)))
