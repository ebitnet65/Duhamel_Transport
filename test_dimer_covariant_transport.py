import numpy as np
import pytest

import Liouville_covariant_transport as lct
import optical_bloch_geometry as obg
from dimer_covariant_geometry import build_dimer_geometry_backend


@pytest.fixture(scope="module")
def geometry():
    if not obg.is_jax_available():
        pytest.skip("JAX is required for the dimer geometry test.")
    return build_dimer_geometry_backend()


def test_dimer_metric_is_nondegenerate_and_connection_is_nontrivial(geometry):
    lam = np.asarray([0.18, 0.04, 0.006])
    metric = geometry.metric(lam)
    gamma = geometry.christoffel(lam)

    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(metric)) > 0.0
    assert np.linalg.norm(gamma) > 1e-4
    np.testing.assert_allclose(gamma, np.swapaxes(gamma, 1, 2), atol=1e-10)


def test_dimer_parallel_transport_preserves_metric_gram_matrix(geometry):
    path = np.asarray(
        [[0.18, 0.04, 0.006], [0.13, 0.055, 0.008], [0.09, 0.025, 0.011]]
    )
    frame0 = lct.metric_orthonormal_frame(geometry.metric(path[0]))
    result = lct.parallel_transport_frame(
        path,
        frame0,
        geometry.christoffel,
        metric_fn=geometry.metric,
        rtol=2e-9,
        atol=2e-11,
    )

    assert result.success, result.message
    assert np.max(result.metric_gram_error) < 2e-7
    np.testing.assert_allclose(
        result.final_frame.T @ geometry.metric(path[-1]) @ result.final_frame,
        np.eye(3),
        atol=2e-7,
    )


def test_dimer_transport_is_reversible(geometry):
    path = np.asarray(
        [[0.18, 0.04, 0.006], [0.13, 0.055, 0.008], [0.09, 0.025, 0.011]]
    )
    frame0 = lct.metric_orthonormal_frame(geometry.metric(path[0]))
    forward = lct.parallel_transport_frame(path, frame0, geometry.christoffel)
    backward = lct.parallel_transport_frame(
        path[::-1], forward.final_frame, geometry.christoffel
    )

    assert forward.success and backward.success
    np.testing.assert_allclose(backward.final_frame, frame0, rtol=2e-7, atol=2e-7)
