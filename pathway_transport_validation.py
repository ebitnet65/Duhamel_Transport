"""Independent invariants and diagnostics for pathway-frame transport.

The routines accept an injected transport callable and do not implement a
connection or transport law.  They compare frames through Gram matrices and
subspaces so that basis-gauge changes are not mistaken for physical effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles

Array = np.ndarray
TransportAdapter = Callable[[Array, Array], Any]
MetricFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class FrameRankDiagnostic:
    rank: int
    expected_rank: int
    singular_values: Array
    threshold: float

    @property
    def rank_lost(self) -> bool:
        return self.rank < self.expected_rank


@dataclass(frozen=True)
class TransportValidation:
    constant_path_error: float
    forward_backward_error: float
    initial_gram_error: float
    final_gram_error: float
    gauge_covariance_error: float
    refinement_errors: Array
    refinement_orders: Array
    final_rank: FrameRankDiagnostic


def _final_frame(result: Any) -> Array:
    if isinstance(result, np.ndarray):
        frame = result
    elif hasattr(result, "final_frame"):
        frame = result.final_frame
    elif isinstance(result, dict) and "final_frame" in result:
        frame = result["final_frame"]
    else:
        raise TypeError("transport result must be an array or expose final_frame")
    frame = np.asarray(frame)
    if frame.ndim == 1:
        frame = frame[:, None]
    if frame.ndim != 2 or not np.all(np.isfinite(frame)):
        raise ValueError("transported frame must be a finite matrix")
    return frame


def frame_rank(frame: Array, *, rtol: float = 1e-10) -> FrameRankDiagnostic:
    """Detect numerical or physical loss of column rank."""

    values = np.linalg.svd(np.asarray(frame), compute_uv=False)
    expected = np.asarray(frame).shape[1]
    scale = float(values[0]) if values.size else 0.0
    threshold = rtol * scale
    return FrameRankDiagnostic(
        rank=int(np.count_nonzero(values > threshold)),
        expected_rank=expected,
        singular_values=values,
        threshold=threshold,
    )


def metric_gram(frame: Array, metric: Array | None = None) -> Array:
    """Return the frame overlap matrix in a supplied ambient metric."""

    basis = np.asarray(frame)
    inner = np.eye(basis.shape[0]) if metric is None else np.asarray(metric)
    if inner.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("metric has incompatible shape")
    return basis.conj().T @ inner @ basis


def metric_subspace_angles(
    first: Array,
    second: Array,
    metric: Array | None = None,
) -> Array:
    """Principal angles after whitening a positive-definite metric."""

    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape[0] != second.shape[0]:
        raise ValueError("frames must share an ambient dimension")
    inner = np.eye(first.shape[0]) if metric is None else np.asarray(metric)
    chol = np.linalg.cholesky(0.5 * (inner + inner.conj().T))
    # G = L L^H for NumPy's lower Cholesky; L^H maps into Euclidean space.
    return subspace_angles(chol.conj().T @ first, chol.conj().T @ second)


def resample_polyline(path: Array, subdivisions: int) -> Array:
    """Subdivide every path segment without changing the geometric polyline."""

    vertices = np.asarray(path, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or subdivisions < 1:
        raise ValueError("invalid path or subdivisions")
    pieces = []
    for index, (start, stop) in enumerate(zip(vertices[:-1], vertices[1:])):
        segment = np.linspace(start, stop, subdivisions + 1)
        pieces.append(segment if index == 0 else segment[1:])
    return np.vstack(pieces)


def validate_transport_adapter(
    transport: TransportAdapter,
    path: Array,
    initial_frame: Array,
    *,
    metric_fn: MetricFunction | None = None,
    refinements: Sequence[int] = (1, 2, 4, 8),
    gauge: Array | None = None,
    rank_rtol: float = 1e-10,
) -> TransportValidation:
    """Evaluate identity, reversibility, overlap, gauge, and refinement tests.

    The adapter must support a two-vertex constant path.  ``gauge`` acts on the
    frame columns and should be nonsingular.  Refinement errors compare each
    result against the finest supplied subdivision using gauge-invariant
    principal angles.
    """

    vertices = np.asarray(path, dtype=float)
    frame = np.asarray(initial_frame)
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("path must contain at least two vertices")
    if frame.ndim != 2:
        raise ValueError("initial_frame must be a matrix")
    constant = np.vstack((vertices[0], vertices[0]))
    constant_final = _final_frame(transport(constant, frame))
    forward = _final_frame(transport(vertices, frame))
    backward = _final_frame(transport(vertices[::-1], forward))

    initial_metric = None if metric_fn is None else metric_fn(vertices[0])
    final_metric = None if metric_fn is None else metric_fn(vertices[-1])
    initial_gram = metric_gram(frame, initial_metric)
    final_gram = metric_gram(forward, final_metric)

    if gauge is None:
        gauge = np.eye(frame.shape[1])
    gauge = np.asarray(gauge)
    if gauge.shape != (frame.shape[1], frame.shape[1]):
        raise ValueError("gauge has incompatible shape")
    gauged_final = _final_frame(transport(vertices, frame @ gauge))

    refinement_frames = [
        _final_frame(transport(resample_polyline(vertices, count), frame))
        for count in refinements
    ]
    reference = refinement_frames[-1]
    errors = np.asarray(
        [
            np.linalg.norm(metric_subspace_angles(value, reference, final_metric))
            for value in refinement_frames
        ]
    )
    orders = []
    for coarse, fine, n_coarse, n_fine in zip(
        errors[:-2], errors[1:-1], refinements[:-2], refinements[1:-1]
    ):
        if coarse > 0.0 and fine > 0.0:
            orders.append(np.log(coarse / fine) / np.log(n_fine / n_coarse))
        else:
            orders.append(np.nan)

    return TransportValidation(
        constant_path_error=float(np.linalg.norm(constant_final - frame)),
        forward_backward_error=float(np.linalg.norm(backward - frame)),
        initial_gram_error=0.0,
        final_gram_error=float(np.linalg.norm(final_gram - initial_gram)),
        gauge_covariance_error=float(np.linalg.norm(gauged_final - forward @ gauge)),
        refinement_errors=errors,
        refinement_orders=np.asarray(orders),
        final_rank=frame_rank(forward, rtol=rank_rtol),
    )


__all__ = [
    "FrameRankDiagnostic",
    "TransportValidation",
    "frame_rank",
    "metric_gram",
    "metric_subspace_angles",
    "resample_polyline",
    "validate_transport_adapter",
]
