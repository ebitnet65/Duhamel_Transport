"""Independent invariant checks against the production pathway-frame API."""

import numpy as np

import Liouville_pathway_parallel_transport as core


def _frame(point):
    x, y = np.asarray(point)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    rotate_01 = np.array(
        [[cx, -sx, 0.0, 0.0], [sx, cx, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    rotate_12 = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cy, -sy, 0.0], [0.0, sy, cy, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    physical = (rotate_01 @ rotate_12)[:, :2].astype(complex)
    phase = np.diag([np.exp(0.4j * x), np.exp(-0.3j * y)])
    return physical @ phase @ np.diag([1.0 + 0.1 * x, 0.8 + 0.1 * y])


def test_core_forward_backward_overlap_gauge_and_refinement():
    path = np.array([[0.0, 0.0], [0.25, 0.1], [0.5, 0.35]])
    frames = np.asarray([_frame(point) for point in path])
    forward = core.parallel_transport_pathway_frames(path, frames=frames)
    backward = core.parallel_transport_pathway_frames(
        path[::-1], frames=frames[::-1]
    )
    identity = np.eye(frames.shape[-1])
    np.testing.assert_allclose(
        backward.final_transport @ forward.final_transport,
        identity,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.max(forward.local_unitarity_errors) < 1e-12
    assert np.max(forward.cumulative_gram_errors) < 1e-12
    assert np.all(forward.frame_ranks == frames.shape[-1])
    assert np.all(forward.overlap_ranks == frames.shape[-1])

    rng = np.random.default_rng(20261021)
    gauges = []
    gauged_frames = []
    for frame in frames:
        trial = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        left, _, right_h = np.linalg.svd(trial)
        gauge = left @ right_h
        gauges.append(gauge)
        gauged_frames.append(frame @ gauge)
    gauged = core.parallel_transport_pathway_frames(
        path, frames=np.asarray(gauged_frames)
    )
    expected = gauges[-1].conj().T @ forward.final_transport @ gauges[0]
    np.testing.assert_allclose(gauged.final_transport, expected, rtol=3e-12, atol=3e-12)

    refinement = core.transport_with_refinement(
        np.array([[0.0, 0.0], [0.5, 0.35]]),
        _frame,
        refinement_factors=(2, 4, 8),
    )
    assert refinement.successive_transport_errors[-1] < refinement.successive_transport_errors[0]
    assert refinement.successive_ambient_frame_errors[-1] < refinement.successive_ambient_frame_errors[0]


def test_closed_loop_holonomy_spectrum_is_local_gauge_invariant():
    loop = np.array(
        [[0.0, 0.0], [0.45, 0.0], [0.45, 0.4], [0.0, 0.4], [0.0, 0.0]]
    )
    frames = np.asarray([_frame(point) for point in loop])
    reference = core.parallel_transport_pathway_frames(loop, frames=frames)

    rng = np.random.default_rng(71)
    gauges = []
    changed_frames = []
    for frame in frames:
        trial = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        left, _, right_h = np.linalg.svd(trial)
        gauge = left @ right_h
        gauges.append(gauge)
        changed_frames.append(frame @ gauge)
    # A closed loop represents one physical base-point gauge, so its final
    # copy must use the same gauge as its initial copy.
    changed_frames[-1] = frames[-1] @ gauges[0]
    changed = core.parallel_transport_pathway_frames(
        loop, frames=np.asarray(changed_frames)
    )
    expected = gauges[0].conj().T @ reference.final_transport @ gauges[0]
    np.testing.assert_allclose(changed.final_transport, expected, rtol=3e-12, atol=3e-12)
    # Trace and determinant are permutation-independent spectral invariants for
    # this two-pathway holonomy.
    np.testing.assert_allclose(
        np.trace(changed.final_transport),
        np.trace(reference.final_transport),
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        np.linalg.det(changed.final_transport),
        np.linalg.det(reference.final_transport),
        rtol=3e-12,
        atol=3e-12,
    )
