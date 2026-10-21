"""Levi-Civita parallel transport on a physical control manifold.

Callers supply Christoffel symbols and, optionally, the metric. A path is
represented by control-space vertices and is linear on each segment. Frame
columns obey dV^a/ds = -Gamma^a_mn dot(lambda)^m V^n.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray
ChristoffelFunction = Callable[[Array], Array]
MetricFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class ParallelTransportResult:
    """Transported frame and diagnostics at every supplied path vertex."""

    path: Array
    frames: Array
    success: bool
    message: str
    nfev: int
    metric_gram_error: Array | None = None

    @property
    def initial_frame(self) -> Array:
        return self.frames[0]

    @property
    def final_frame(self) -> Array:
        return self.frames[-1]


def metric_orthonormal_frame(metric: Array) -> Array:
    """Construct E with E.T @ G @ E = I for a positive-definite metric."""

    metric = np.asarray(metric, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be a square matrix")
    chol = np.linalg.cholesky(0.5 * (metric + metric.T))
    return np.linalg.solve(chol.T, np.eye(metric.shape[0]))


def _validate_path(path: Array) -> Array:
    vertices = np.asarray(path, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("path must have shape (n_vertices, n_controls), n_vertices >= 2")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("path contains non-finite controls")
    if np.any(np.linalg.norm(np.diff(vertices, axis=0), axis=1) == 0.0):
        raise ValueError("adjacent path vertices must be distinct")
    return vertices


def parallel_transport_frame(
    path: Array,
    initial_frame: Array,
    christoffel_fn: ChristoffelFunction,
    *,
    metric_fn: MetricFunction | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    method: str = "DOP853",
    max_step: float = np.inf,
) -> ParallelTransportResult:
    """Parallel transport vector columns along a piecewise-linear path."""

    vertices = _validate_path(path)
    dimension = vertices.shape[1]
    frame = np.asarray(initial_frame, dtype=float)
    if frame.ndim == 1:
        frame = frame[:, None]
    if frame.ndim != 2 or frame.shape[0] != dimension:
        raise ValueError("initial_frame must have shape (n_controls, n_vectors)")
    if not np.all(np.isfinite(frame)):
        raise ValueError("initial_frame contains non-finite values")

    frames = [frame.copy()]
    total_nfev = 0
    messages: list[str] = []
    success = True

    for segment, (start, stop) in enumerate(zip(vertices[:-1], vertices[1:])):
        velocity = stop - start

        def rhs(t, flat_frame):
            lam = start + t * velocity
            gamma = np.asarray(christoffel_fn(lam), dtype=float)
            if gamma.shape != (dimension, dimension, dimension):
                raise ValueError(
                    "christoffel_fn must return shape "
                    f"({dimension}, {dimension}, {dimension})"
                )
            connection = np.einsum("amn,m->an", gamma, velocity)
            current = flat_frame.reshape(frame.shape)
            return (-connection @ current).reshape((-1,))

        solution = solve_ivp(
            rhs,
            (0.0, 1.0),
            frames[-1].reshape((-1,)),
            method=method,
            t_eval=[1.0],
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        total_nfev += solution.nfev
        messages.append(f"segment {segment}: {solution.message}")
        if not solution.success:
            success = False
            break
        frames.append(solution.y[:, -1].reshape(frame.shape))

    frame_history = np.asarray(frames)
    gram_error = None
    if metric_fn is not None:
        initial_metric = np.asarray(metric_fn(vertices[0]), dtype=float)
        initial_gram = frame_history[0].T @ initial_metric @ frame_history[0]
        gram_error = np.asarray(
            [
                np.linalg.norm(
                    transported.T @ np.asarray(metric_fn(lam), dtype=float) @ transported
                    - initial_gram,
                    ord="fro",
                )
                for lam, transported in zip(vertices[: len(frame_history)], frame_history)
            ]
        )

    return ParallelTransportResult(
        path=vertices[: len(frame_history)].copy(),
        frames=frame_history,
        success=success,
        message="; ".join(messages),
        nfev=total_nfev,
        metric_gram_error=gram_error,
    )


def parallel_transport_vector(
    path: Array,
    initial_vector: Array,
    christoffel_fn: ChristoffelFunction,
    **kwargs,
) -> ParallelTransportResult:
    """Convenience wrapper for transporting one tangent vector."""

    return parallel_transport_frame(path, initial_vector, christoffel_fn, **kwargs)


__all__ = [
    "ParallelTransportResult",
    "metric_orthonormal_frame",
    "parallel_transport_frame",
    "parallel_transport_vector",
]
