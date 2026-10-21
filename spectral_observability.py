"""Coordinate-covariant observability geometry for spectral inverse problems.

The observable may be real or complex and may have any array shape.  Complex
observations are represented by stacking their real and imaginary parts, so
all information matrices are real symmetric matrices on the control space.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

Array = np.ndarray


@dataclass(frozen=True)
class ObservabilityModes:
    """Generalized eigenmodes of ``I v = chi G v`` and diagnostics."""

    eigenvalues: Array
    eigenvectors: Array
    effective_rank: int
    condition_number: float
    threshold: float

    @property
    def dark_mask(self) -> Array:
        """Boolean mask selecting modes below the numerical rank threshold."""

        return self.eigenvalues <= self.threshold


def real_observation(observation: Array) -> Array:
    """Flatten an observation into a real residual vector.

    For complex data the ordering is ``[Re(y).ravel(), Im(y).ravel()]``.
    """

    values = np.asarray(observation)
    if np.iscomplexobj(values):
        return np.concatenate((values.real.ravel(), values.imag.ravel()))
    return np.asarray(values, dtype=float).ravel()


def real_jacobian(jacobian: Array) -> Array:
    """Convert an observable Jacobian to a real residual Jacobian.

    The final input axis is interpreted as the control axis.  All preceding
    observable axes are flattened.  A complex Jacobian is stacked in the same
    real-then-imaginary convention as :func:`real_observation`.
    """

    derivative = np.asarray(jacobian)
    if derivative.ndim < 1:
        raise ValueError("jacobian must have at least one axis")
    n_controls = derivative.shape[-1]
    flattened = derivative.reshape((-1, n_controls))
    if np.iscomplexobj(flattened):
        return np.vstack((flattened.real, flattened.imag))
    return np.asarray(flattened, dtype=float)


def _weight_operator(weights: Array | float | None, observation_shape: tuple[int, ...], n_rows: int) -> Array:
    if weights is None:
        return np.eye(n_rows)
    weight = np.asarray(weights, dtype=float)
    if weight.ndim == 0:
        return float(weight) * np.eye(n_rows)
    if weight.shape == (n_rows, n_rows):
        return weight
    if weight.size == n_rows:
        return np.diag(weight.ravel())

    n_observations = int(np.prod(observation_shape, dtype=int))
    if n_rows == 2 * n_observations and weight.size == n_observations:
        entries = weight.ravel()
        return np.diag(np.concatenate((entries, entries)))
    raise ValueError(
        "weights must be scalar, observation-shaped, residual-length, or a "
        "square residual-space operator"
    )


def weighted_residual_jacobian(
    jacobian: Array,
    *,
    weights: Array | float | None = None,
    observation_shape: tuple[int, ...] | None = None,
) -> Array:
    """Return ``W J`` in a real residual representation.

    ``jacobian`` has shape ``observation_shape + (n_controls,)``.  For a
    complex Jacobian, observation-shaped scalar weights are applied equally to
    real and imaginary residuals.
    """

    raw = np.asarray(jacobian)
    if raw.ndim < 1:
        raise ValueError("jacobian must have at least one axis")
    inferred_shape = raw.shape[:-1]
    shape = inferred_shape if observation_shape is None else tuple(observation_shape)
    if int(np.prod(shape, dtype=int)) != int(np.prod(inferred_shape, dtype=int)):
        raise ValueError("observation_shape is incompatible with jacobian")
    real_derivative = real_jacobian(raw)
    operator = _weight_operator(weights, shape, real_derivative.shape[0])
    return operator @ real_derivative


def observable_jacobian(
    observable_fn: Callable[[Array], Array],
    parameters: Array,
    *,
    jacobian_fn: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array]:
    """Evaluate an observable and its real Jacobian.

    A supplied ``jacobian_fn`` may return a NumPy or JAX array and may be
    complex.  If omitted, JAX automatic differentiation is used on the real
    residual representation.
    """

    point = np.asarray(parameters, dtype=float)
    if point.ndim != 1:
        raise ValueError("parameters must be a one-dimensional vector")
    observation = np.asarray(observable_fn(point))
    if jacobian_fn is not None:
        derivative = np.asarray(jacobian_fn(point))
        expected_shape = observation.shape + point.shape
        if derivative.shape != expected_shape:
            raise ValueError(f"jacobian_fn returned {derivative.shape}, expected {expected_shape}")
        return observation, real_jacobian(derivative)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised without JAX
        raise ImportError("JAX is required when jacobian_fn is not supplied") from exc

    def real_map(control):
        value = jnp.asarray(observable_fn(control))
        flat = jnp.ravel(value)
        if jnp.issubdtype(flat.dtype, jnp.complexfloating):
            return jnp.concatenate((jnp.real(flat), jnp.imag(flat)))
        return flat

    derivative = np.asarray(jax.jacrev(real_map)(jnp.asarray(point)))
    return observation, derivative


def information_metric(
    jacobian: Array,
    *,
    weights: Array | float | None = None,
    observation_shape: tuple[int, ...] | None = None,
    assume_real_residual: bool = False,
) -> Array:
    """Compute the spectral information metric ``I = J.T W.T W J``."""

    if assume_real_residual:
        derivative = np.asarray(jacobian, dtype=float)
        if derivative.ndim != 2:
            raise ValueError("a real residual Jacobian must be a matrix")
        shape = (derivative.shape[0],) if observation_shape is None else observation_shape
        operator = _weight_operator(weights, tuple(shape), derivative.shape[0])
        weighted = operator @ derivative
    else:
        weighted = weighted_residual_jacobian(
            jacobian, weights=weights, observation_shape=observation_shape
        )
    metric = weighted.T @ weighted
    return 0.5 * (metric + metric.T)


def generalized_observability_modes(
    information: Array,
    physical_metric: Array,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> ObservabilityModes:
    """Solve ``I v = chi G v`` with descending, G-orthonormal modes."""

    info = np.asarray(information, dtype=float)
    physical = np.asarray(physical_metric, dtype=float)
    if info.ndim != 2 or info.shape[0] != info.shape[1]:
        raise ValueError("information must be square")
    if physical.shape != info.shape:
        raise ValueError("physical_metric must match information")
    info = 0.5 * (info + info.T)
    physical = 0.5 * (physical + physical.T)
    try:
        np.linalg.cholesky(physical)
    except np.linalg.LinAlgError as exc:
        raise ValueError("physical_metric must be positive definite") from exc

    values, vectors = eigh(info, physical, check_finite=True)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    scale = max(float(values[0]), 0.0) if values.size else 0.0
    threshold = max(float(atol), float(rtol) * scale)
    # Roundoff can produce tiny negative values for positive-semidefinite I.
    tolerance = max(threshold, np.finfo(float).eps * max(scale, 1.0) * info.shape[0])
    if np.any(values < -tolerance):
        raise ValueError("information must be positive semidefinite")
    values = np.where(np.abs(values) <= tolerance, 0.0, values)
    visible = values > threshold
    rank = int(np.count_nonzero(visible))
    condition = float(values[0] / values[visible][-1]) if rank else np.inf
    return ObservabilityModes(values, vectors, rank, condition, threshold)


def joint_information_metric(information_metrics: Iterable[Array]) -> Array:
    """Add independent information contributions, e.g. across waiting times."""

    metrics = [np.asarray(metric, dtype=float) for metric in information_metrics]
    if not metrics:
        raise ValueError("at least one information metric is required")
    shape = metrics[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("information metrics must be square")
    if any(metric.shape != shape for metric in metrics):
        raise ValueError("all information metrics must have the same shape")
    total = np.sum(metrics, axis=0)
    return 0.5 * (total + total.T)


def information_across_waiting_times(
    jacobians: Sequence[Array],
    *,
    weights: Sequence[Array | float | None] | Array | float | None = None,
) -> tuple[Array, list[Array]]:
    """Compute individual and joint information for waiting-time Jacobians."""

    if isinstance(weights, (list, tuple)):
        if len(weights) != len(jacobians):
            raise ValueError("weights must have one entry per waiting time")
        weight_list = list(weights)
    else:
        weight_list = [weights] * len(jacobians)
    individual = [
        information_metric(jacobian, weights=weight)
        for jacobian, weight in zip(jacobians, weight_list, strict=True)
    ]
    return joint_information_metric(individual), individual


__all__ = [
    "ObservabilityModes",
    "generalized_observability_modes",
    "information_across_waiting_times",
    "information_metric",
    "joint_information_metric",
    "observable_jacobian",
    "real_jacobian",
    "real_observation",
    "weighted_residual_jacobian",
]
