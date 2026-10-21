import numpy as np
from scipy.linalg import expm

from pathway_transport_validation import (
    frame_rank,
    metric_subspace_angles,
    validate_transport_adapter,
)


def _constant_connection_transport(generators):
    """Analytically compose segment transports for controlled validation."""

    generators = np.asarray(generators)

    def transport(path, frame):
        result = np.asarray(frame)
        for displacement in np.diff(np.asarray(path), axis=0):
            connection = np.einsum("i,ijk->jk", displacement, generators)
            result = expm(-connection) @ result
        return result

    return transport


def test_identity_reverse_metric_gauge_and_refinement_invariants():
    generator = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    transport = _constant_connection_transport([generator])
    path = np.array([[0.0], [0.4], [1.1]])
    frame = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    gauge = np.array([[1.0, 0.3], [-0.2, 1.1]])
    result = validate_transport_adapter(
        transport,
        path,
        frame,
        metric_fn=lambda _q: np.eye(3),
        gauge=gauge,
    )
    assert result.constant_path_error < 1e-14
    assert result.forward_backward_error < 1e-14
    assert result.final_gram_error < 1e-14
    assert result.gauge_covariance_error < 1e-14
    assert np.max(result.refinement_errors) < 1e-14
    assert not result.final_rank.rank_lost


def test_rank_loss_is_detected_without_confusing_small_norm_with_rank_loss():
    healthy_small = frame_rank(1e-12 * np.eye(3, 2))
    deficient = frame_rank(np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))
    assert not healthy_small.rank_lost
    assert deficient.rank_lost
    assert deficient.rank == 1


def test_internal_basis_gauge_does_not_change_subspace():
    frame = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    gauge = np.array([[2.0, -0.4], [0.3, 0.7]])
    angles = metric_subspace_angles(frame, frame @ gauge, np.diag([2.0, 3.0, 4.0]))
    assert np.max(angles) < 1e-14


def test_noncommuting_connection_produces_genuine_path_dependent_subspace():
    rotation_xy = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotation_yz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    transport = _constant_connection_transport([rotation_xy, rotation_yz])
    frame = np.array([[1.0], [0.0], [0.0]])
    xy_path = np.array([[0.0, 0.0], [0.7, 0.0], [0.7, 0.8]])
    yx_path = np.array([[0.0, 0.0], [0.0, 0.8], [0.7, 0.8]])
    final_xy = transport(xy_path, frame)
    final_yx = transport(yx_path, frame)
    angle = metric_subspace_angles(final_xy, final_yx)[0]
    assert angle > 0.2
    # A change of the one-column basis gauge cannot remove this angle.
    assert np.allclose(
        metric_subspace_angles(2.7 * final_xy, -0.4 * final_yx)[0], angle
    )


def test_coordinate_representation_change_preserves_physical_result():
    generator = np.array([[0.0, -0.6], [0.6, 0.0]])
    base_transport = _constant_connection_transport([generator])
    path = np.array([[0.0], [0.3], [1.0]])
    frame = np.array([[1.0], [0.2]])
    transform_start = np.array([[1.3, 0.2], [0.1, 0.8]])
    transform_stop = np.array([[0.9, -0.1], [0.2, 1.2]])

    physical_final = base_transport(path, frame)
    # Passive representation change: coordinates are S(q) times physical
    # vectors, so the endpoint map is S(q1) T S(q0)^-1.
    represented_initial = transform_start @ frame
    represented_final = (
        transform_stop
        @ base_transport(path, np.linalg.solve(transform_start, represented_initial))
    )
    recovered_physical = np.linalg.solve(transform_stop, represented_final)
    assert np.allclose(recovered_physical, physical_final)
