import numpy as np
import pytest

from spectral_observability import (
    generalized_observability_modes,
    information_across_waiting_times,
    information_metric,
    joint_information_metric,
    observable_jacobian,
    real_observation,
    weighted_residual_jacobian,
)


def test_complex_observation_and_weighted_information():
    jacobian = np.array([[1.0 + 2.0j, 3.0 - 1.0j], [2.0j, 1.0 + 0.5j]])
    weighted = weighted_residual_jacobian(jacobian, weights=np.array([2.0, 0.5]))
    expected_real = np.vstack((jacobian.real, jacobian.imag))
    expected_weights = np.diag([2.0, 0.5, 2.0, 0.5])
    assert np.allclose(weighted, expected_weights @ expected_real)
    assert np.allclose(
        information_metric(jacobian, weights=np.array([2.0, 0.5])),
        weighted.T @ weighted,
    )
    assert np.allclose(real_observation(np.array([1 + 2j, 3 - 4j])), [1, 3, 2, -4])


def test_supplied_jax_style_jacobian_is_accepted():
    point = np.array([0.3, -0.2])

    def observable(q):
        return np.array([q[0] + 1j * q[1], q[0] * q[1]])

    def derivative(q):
        return np.array([[1.0 + 0.0j, 1.0j], [q[1], q[0]]])

    observation, jacobian = observable_jacobian(observable, point, jacobian_fn=derivative)
    assert observation.shape == (2,)
    assert jacobian.shape == (4, 2)
    assert np.allclose(jacobian, np.vstack((derivative(point).real, derivative(point).imag)))


def test_autodiff_real_and_complex_observables():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    point = np.array([0.4, -0.3])

    def observable(q):
        return jnp.array([q[0] ** 2 + 1j * q[1], q[0] * q[1]])

    _, jacobian = observable_jacobian(observable, point)
    expected = np.array([[0.8, 0.0], [-0.3, 0.4], [0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(jacobian, expected)


def test_generalized_modes_are_g_orthonormal_and_detect_dark_mode():
    physical = np.diag([2.0, 3.0, 5.0])
    information = np.diag([8.0, 3.0, 0.0])
    modes = generalized_observability_modes(information, physical)
    assert np.allclose(modes.eigenvalues, [4.0, 1.0, 0.0])
    assert modes.effective_rank == 2
    assert modes.condition_number == pytest.approx(4.0)
    assert np.array_equal(modes.dark_mask, [False, False, True])
    assert np.allclose(modes.eigenvectors.T @ physical @ modes.eigenvectors, np.eye(3))
    for value, vector in zip(modes.eigenvalues, modes.eigenvectors.T, strict=True):
        assert np.allclose(information @ vector, value * physical @ vector)


def test_coordinate_reparameterization_preserves_generalized_spectrum():
    jacobian_q = np.array([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [2.0, 0.3, 0.7]])
    physical_q = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.2]])
    # dq = A dtheta for an invertible, non-orthogonal coordinate change.
    transform = np.array([[1.2, 0.2, 0.0], [0.1, 0.8, 0.3], [0.0, -0.2, 1.1]])
    information_q = information_metric(jacobian_q)
    information_theta = information_metric(jacobian_q @ transform)
    physical_theta = transform.T @ physical_q @ transform
    modes_q = generalized_observability_modes(information_q, physical_q)
    modes_theta = generalized_observability_modes(information_theta, physical_theta)
    assert np.allclose(modes_theta.eigenvalues, modes_q.eigenvalues)
    # Each nondegenerate theta mode maps to the same q-space line (up to sign).
    mapped = transform @ modes_theta.eigenvectors
    overlaps = np.abs(modes_q.eigenvectors.T @ physical_q @ mapped)
    assert np.allclose(overlaps, np.eye(3), atol=1e-10)


def test_joint_waiting_times_close_complementary_dark_directions():
    jacobian_short = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    jacobian_long = np.array([[0.0, 0.0, 2.0]])
    joint, individual = information_across_waiting_times(
        [jacobian_short, jacobian_long], weights=[1.0, 0.5]
    )
    assert np.allclose(individual[0], np.diag([1.0, 1.0, 0.0]))
    assert np.allclose(individual[1], np.diag([0.0, 0.0, 1.0]))
    assert np.allclose(joint, np.eye(3))
    assert generalized_observability_modes(individual[0], np.eye(3)).effective_rank == 2
    assert generalized_observability_modes(joint, np.eye(3)).effective_rank == 3
    assert np.allclose(joint_information_metric(individual), joint)


def test_invalid_physical_metric_is_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        generalized_observability_modes(np.eye(2), np.diag([1.0, 0.0]))
