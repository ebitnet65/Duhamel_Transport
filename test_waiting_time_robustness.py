import json

import numpy as np
import pytest

from waiting_time_robustness import analyze_waiting_time_robustness


def test_fixed_endpoint_recovery_and_complementary_waiting_time_information():
    truth = np.array([0.4, -0.2, 0.3])

    def observation(q, t2):
        matrix = np.array([[1.0, 0.0, t2], [0.0, 1.0, 1.0 - t2]])
        return matrix @ q

    def jacobian(_q, t2):
        return np.array([[1.0, 0.0, t2], [0.0, 1.0, 1.0 - t2]])

    # This deterministic stand-in tests the analyzer's optimizer adapter; the
    # production benchmark injects the common endpoint inversion routine.
    recover = lambda _t2, _data: {"endpoint": truth.copy()}
    result = analyze_waiting_time_robustness(
        truth,
        [0.0, 1.0],
        observation,
        recover,
        lambda _q: np.diag([2.0, 3.0, 4.0]),
        jacobian_fn=jacobian,
    )

    assert all(item.endpoint_drift == 0.0 for item in result.diagnostics)
    assert all(item.residual_norm == 0.0 for item in result.diagnostics)
    assert all(item.generalized_effective_rank == 2 for item in result.diagnostics)
    assert result.joint_effective_rank == 3
    assert np.all(np.linalg.eigvalsh(result.joint_information_metric) > 0.0)
    # Machine-readable conversion must contain no NumPy-only objects.
    payload = result.to_dict()
    assert json.loads(json.dumps(payload))["truth_endpoint"] == truth.tolist()


def test_complex_observation_autodiff_path():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    truth = np.array([0.3, 0.2])

    def observation(q, t2):
        return jnp.array([q[0] + 1j * t2 * q[1], q[0] * q[1]])

    result = analyze_waiting_time_robustness(
        truth,
        [0.5, 1.5],
        observation,
        lambda _t2, _data: truth,
        lambda _q: np.eye(2),
    )
    assert result.joint_effective_rank == 2
    assert all(item.residual_norm == 0.0 for item in result.diagnostics)
    assert not np.allclose(
        result.diagnostics[0].information_metric,
        result.diagnostics[1].information_metric,
    )


def test_invalid_recovered_endpoint_is_rejected():
    with pytest.raises(ValueError, match="invalid shape"):
        analyze_waiting_time_robustness(
            np.array([1.0, 2.0]),
            [1.0],
            lambda q, _t2: q,
            lambda _t2, _data: np.array([1.0]),
            lambda _q: np.eye(2),
            jacobian_fn=lambda _q, _t2: np.eye(2),
        )
