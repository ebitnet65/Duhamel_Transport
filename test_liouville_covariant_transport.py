import numpy as np
import pytest

import Liouville_covariant_transport as lct
import optical_bloch_geometry as obg


def _metric(lam):
    return obg.symbolic_geometry(lam).metric


def _christoffel(lam):
    return obg.symbolic_geometry(lam).christoffel


def test_metric_orthonormal_frame():
    lam = np.asarray([0.35, 0.12, 0.08])
    metric = _metric(lam)
    frame = lct.metric_orthonormal_frame(metric)
    np.testing.assert_allclose(frame.T @ metric @ frame, np.eye(3), atol=1e-13)


def test_parallel_transport_preserves_metric_inner_products():
    path = np.asarray(
        [[0.35, 0.12, 0.08], [0.39, 0.08, 0.095], [0.44, -0.02, 0.11]]
    )
    frame0 = lct.metric_orthonormal_frame(_metric(path[0]))
    result = lct.parallel_transport_frame(
        path, frame0, _christoffel, metric_fn=_metric, rtol=2e-10, atol=2e-12
    )

    assert result.success, result.message
    assert result.frames.shape == (len(path), 3, 3)
    assert np.max(result.metric_gram_error) < 2e-8
    np.testing.assert_allclose(
        result.final_frame.T @ _metric(path[-1]) @ result.final_frame,
        np.eye(3),
        atol=2e-8,
    )


def test_reverse_path_recovers_initial_frame():
    path = np.asarray(
        [[0.35, 0.12, 0.08], [0.39, 0.08, 0.095], [0.44, -0.02, 0.11]]
    )
    frame0 = lct.metric_orthonormal_frame(_metric(path[0]))
    forward = lct.parallel_transport_frame(path, frame0, _christoffel)
    backward = lct.parallel_transport_frame(path[::-1], forward.final_frame, _christoffel)

    assert forward.success and backward.success
    np.testing.assert_allclose(backward.final_frame, frame0, rtol=2e-8, atol=2e-8)


def test_liouvillian_autodiff_christoffel_drives_transport_when_jax_available():
    if not obg.is_jax_available():
        pytest.skip("JAX is required for the Liouvillian-autodiff transport test.")

    path = np.asarray([[0.35, 0.12, 0.08], [0.37, 0.10, 0.09]])
    frame0 = lct.metric_orthonormal_frame(_metric(path[0]))

    def autodiff_christoffel(lam):
        return obg.jax_liouvillian_geometry(lam)["christoffel"]

    symbolic = lct.parallel_transport_frame(path, frame0, _christoffel)
    autodiff = lct.parallel_transport_frame(path, frame0, autodiff_christoffel)

    assert autodiff.success
    np.testing.assert_allclose(
        autodiff.final_frame, symbolic.final_frame, rtol=2e-9, atol=2e-9
    )
