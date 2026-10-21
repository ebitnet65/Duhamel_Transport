"""Discrete geodesics for a model-supplied physical control metric."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray
MetricFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class GeodesicResult:
    """Minimum-energy discrete path and optimization diagnostics."""

    path: Array
    energy: float
    initial_energy: float
    success: bool
    message: str
    iterations: int
    function_evaluations: int
    metric_determinants: Array
    metric_condition_numbers: Array

    @property
    def relative_energy_reduction(self) -> float:
        if self.initial_energy == 0.0:
            return 0.0
        return (self.initial_energy - self.energy) / self.initial_energy


def linear_path(start: Array, stop: Array, n_vertices: int) -> Array:
    """Return an evenly parameterized coordinate-space straight path."""

    if n_vertices < 2:
        raise ValueError("n_vertices must be at least two")
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    if start.shape != stop.shape or start.ndim != 1:
        raise ValueError("start and stop must be equal-length vectors")
    return np.linspace(start, stop, n_vertices)


def discrete_path_energy(path: Array, metric_fn: MetricFunction) -> float:
    """Midpoint approximation to 1/2 integral dot(lam).T G dot(lam) ds."""

    vertices = np.asarray(path, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("path must have shape (n_vertices, n_controls)")
    segments = np.diff(vertices, axis=0)
    midpoints = 0.5 * (vertices[:-1] + vertices[1:])
    n_segments = len(segments)
    total = 0.0
    for displacement, midpoint in zip(segments, midpoints):
        metric = np.asarray(metric_fn(midpoint), dtype=float)
        if metric.shape != (vertices.shape[1], vertices.shape[1]):
            raise ValueError("metric_fn returned a matrix with the wrong shape")
        total += float(displacement @ metric @ displacement)
    return 0.5 * n_segments * total


def solve_discrete_geodesic(
    start: Array,
    stop: Array,
    metric_fn: MetricFunction,
    *,
    n_vertices: int = 9,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    method: str = "L-BFGS-B",
    options: dict | None = None,
) -> GeodesicResult:
    """Minimize discrete path energy over interior vertices at fixed endpoints."""

    initial_path = linear_path(start, stop, n_vertices)
    dimension = initial_path.shape[1]
    if bounds is not None and len(bounds) != dimension:
        raise ValueError("bounds must contain one pair per control")
    optimizer_bounds = None
    if bounds is not None:
        for name, endpoint in (("start", initial_path[0]), ("stop", initial_path[-1])):
            for index, (value, bound) in enumerate(zip(endpoint, bounds, strict=True)):
                lower, upper = bound
                if lower is not None and value < lower:
                    raise ValueError(f"{name}[{index}] lies below its lower bound")
                if upper is not None and value > upper:
                    raise ValueError(f"{name}[{index}] lies above its upper bound")
        optimizer_bounds = list(bounds) * (n_vertices - 2)

    def unpack(interior_flat):
        path = initial_path.copy()
        path[1:-1] = np.asarray(interior_flat).reshape((n_vertices - 2, dimension))
        return path

    def objective(interior_flat):
        return discrete_path_energy(unpack(interior_flat), metric_fn)

    initial_energy = discrete_path_energy(initial_path, metric_fn)
    result = minimize(
        objective,
        initial_path[1:-1].reshape((-1,)),
        method=method,
        bounds=optimizer_bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, **(options or {})},
    )
    path = unpack(result.x)
    metrics = [np.asarray(metric_fn(vertex), dtype=float) for vertex in path]
    return GeodesicResult(
        path=path,
        energy=float(result.fun),
        initial_energy=initial_energy,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        metric_determinants=np.asarray([np.linalg.det(metric) for metric in metrics]),
        metric_condition_numbers=np.asarray([np.linalg.cond(metric) for metric in metrics]),
    )


__all__ = [
    "GeodesicResult",
    "discrete_path_energy",
    "linear_path",
    "solve_discrete_geodesic",
]
