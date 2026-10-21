import numpy as np
import pytest

import Liouville_pathway_parallel_transport as lppt
import Liouville_transport_general as general
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map


def smooth_frame(point):
    """Two-plane in C^4 with nontrivial rotation and internal phase gauge."""

    x = float(np.asarray(point)[0])
    physical = np.column_stack(
        (
            [np.cos(x), 0.0, np.sin(x), 0.0],
            [0.0, np.cos(0.6 * x), 0.0, np.sin(0.6 * x)],
        )
    ).astype(complex)
    gauge = np.asarray(
        [
            [np.exp(0.3j * x) * np.cos(0.2 * x), -np.sin(0.2 * x)],
            [np.sin(0.2 * x), np.exp(-0.3j * x) * np.cos(0.2 * x)],
        ],
        dtype=complex,
    )
    # A positive nonunitary column scaling tests stable polar normalization.
    return physical @ gauge @ np.diag([1.0 + 0.2 * x, 0.8 + 0.1 * x])


def curved_frame(point):
    """Noncommuting ambient rotations give a nontrivial discrete connection."""

    x = float(np.asarray(point)[0])
    c1, s1 = np.cos(x), np.sin(x)
    c2, s2 = np.cos(0.8 * x * x), np.sin(0.8 * x * x)
    rotate_01 = np.asarray(
        [[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]]
    )
    rotate_12 = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]]
    )
    return (rotate_01 @ rotate_12)[:, :2].astype(complex)


def test_transport_is_isometric_and_reports_subspace_geometry():
    path = np.linspace(0.0, 0.7, 8)[:, None]
    result = lppt.parallel_transport_pathway_frames(path, frame_fn=smooth_frame)
    assert result.local_transport_maps.shape == (7, 2, 2)
    assert result.principal_angles.shape == (7, 2)
    assert np.all(result.frame_ranks == 2)
    assert np.all(result.overlap_ranks == 2)
    assert np.max(result.local_unitarity_errors) < 1e-12
    assert np.max(result.cumulative_gram_errors) < 1e-12
    assert np.all(result.principal_angles > 0.0)
    assert np.all(result.principal_angles < np.pi / 2.0)
    # Orthogonal projection contracts as the physical subspace rotates.
    assert result.projection_gram_errors[-1] > 1e-4


def test_transport_is_covariant_under_arbitrary_local_unitary_gauges():
    path = np.linspace(0.0, 0.65, 7)[:, None]
    frames = np.asarray([smooth_frame(point) for point in path])
    reference = lppt.parallel_transport_pathway_frames(path, frames=frames)
    gauges = []
    transformed = []
    for index, frame in enumerate(frames):
        theta = 0.17 * (index + 1)
        phase = 0.11 * index
        gauge = np.asarray(
            [
                [np.exp(1j * phase) * np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.exp(-1j * phase) * np.cos(theta)],
            ],
            dtype=complex,
        )
        # Project to exactly unitary because the displayed ansatz has mixed phases.
        u, _, vh = np.linalg.svd(gauge)
        gauge = u @ vh
        gauges.append(gauge)
        transformed.append(frame @ gauge)
    changed = lppt.parallel_transport_pathway_frames(path, frames=np.asarray(transformed))

    expected = gauges[-1].conj().T @ reference.final_transport @ gauges[0]
    np.testing.assert_allclose(changed.final_transport, expected, rtol=2e-12, atol=2e-12)
    initial_coefficients = np.asarray([0.7 + 0.2j, -0.3 + 0.1j])
    reference_ambient = (
        reference.orthonormal_frames[-1]
        @ reference.final_transport
        @ initial_coefficients
    )
    changed_initial = gauges[0].conj().T @ initial_coefficients
    changed_ambient = (
        changed.orthonormal_frames[-1]
        @ changed.final_transport
        @ changed_initial
    )
    np.testing.assert_allclose(changed_ambient, reference_ambient, rtol=2e-12, atol=2e-12)


def test_refinement_converges_for_smooth_rotating_subspace():
    refinement = lppt.transport_with_refinement(
        np.asarray([[0.0], [0.8]]),
        curved_frame,
        refinement_factors=(2, 4, 8),
    )
    assert refinement.successive_transport_errors.shape == (2,)
    assert refinement.successive_ambient_frame_errors.shape == (2,)
    assert refinement.successive_transport_errors[-1] < refinement.successive_transport_errors[0]
    assert refinement.successive_ambient_frame_errors[-1] < refinement.successive_ambient_frame_errors[0]


def test_rank_loss_and_orthogonal_adjacent_subspaces_are_rejected():
    path = np.asarray([[0.0], [1.0]])
    rank_deficient = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=complex,
    )
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        lppt.parallel_transport_pathway_frames(path, frames=rank_deficient)

    orthogonal = np.asarray(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ],
        dtype=complex,
    )
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient overlap"):
        lppt.parallel_transport_pathway_frames(path, frames=orthogonal)


def test_existing_dimer_pathway_frames_use_ordered_model_adapter():
    model = exciton_dimer_model()
    parameter_map = exciton_dimer_parameter_map(
        general.ControlSpec("J_12", reference_value=0.02, scale=0.01)
    )
    frequencies = np.linspace(0.95, 1.25, 3)
    candidates = general.canonical_pathways(("RP",))
    pathways, _ = general.independent_canonical_pathways(
        model,
        parameter_map,
        frequencies,
        candidates,
        stage="after_U2",
        tau2=0.5,
        eta=0.03,
    )
    frame_fn = lppt.build_liouville_pathway_frame_function(
        model,
        parameter_map,
        frequencies,
        pathways,
        stage="after_U2",
        tau2=0.5,
        eta=0.03,
    )
    path = np.asarray([[0.0], [0.1], [0.2]])
    result = lppt.parallel_transport_pathway_frames(path, frame_fn=frame_fn)
    assert result.raw_frames.shape == (3, 48, len(pathways))
    assert len(pathways) == 2
    assert np.all(result.frame_ranks == 2)
    assert np.max(result.cumulative_gram_errors) < 1e-12
    assert np.all(result.principal_angles > 0.0)
