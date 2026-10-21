"""Waiting-time robustness diagnostics for endpoint spectral inversion.

This module deliberately receives endpoint recovery as an injected callable.
It therefore analyzes a common inversion implementation without duplicating or
coupling to its optimizer.  The truth observable is always an endpoint
property; waiting time changes the observation and its local information, not
the synthetic truth endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spectral_observability import (
    generalized_observability_modes,
    information_metric,
    joint_information_metric,
)

Array = np.ndarray


@dataclass(frozen=True)
class WaitingTimeDiagnostic:
    """Endpoint and local-observability diagnostics at one waiting time."""

    waiting_time: float
    endpoint: Array
    endpoint_drift: float
    residual_norm: float
    relative_residual_norm: float
    jacobian_singular_values: Array
    jacobian_condition_number: float
    generalized_eigenvalues: Array
    generalized_effective_rank: int
    generalized_condition_number: float
    information_metric: Array


@dataclass(frozen=True)
class WaitingTimeRobustnessResult:
    """Independent fits plus the combined multi-waiting-time information."""

    truth_endpoint: Array
    diagnostics: tuple[WaitingTimeDiagnostic, ...]
    joint_information_metric: Array
    joint_generalized_eigenvalues: Array
    joint_effective_rank: int
    joint_condition_number: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def _endpoint_from_recovery(result: Any) -> Array:
    """Adapt a recovery result without imposing an optimizer-specific class."""

    if isinstance(result, np.ndarray) or isinstance(result, (list, tuple)):
        return np.asarray(result, dtype=float)
    for name in ("endpoint", "parameters", "x"):
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float)
    if isinstance(result, dict):
        for name in ("endpoint", "parameters", "x"):
            if name in result:
                return np.asarray(result[name], dtype=float)
    raise TypeError("recovery result must expose endpoint, parameters, or x")


def _real_vector(values: Array) -> Array:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        return np.concatenate((array.real.ravel(), array.imag.ravel()))
    return np.asarray(array, dtype=float).ravel()


def _autodiff_jacobian(observable_fn, endpoint: Array, waiting_time: float) -> Array:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - only without JAX
        raise ImportError("JAX is required when jacobian_fn is not supplied") from exc

    def real_observable(q):
        values = jnp.ravel(jnp.asarray(observable_fn(q, waiting_time)))
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            return jnp.concatenate((jnp.real(values), jnp.imag(values)))
        return values

    return np.asarray(jax.jacfwd(real_observable)(jnp.asarray(endpoint)))


def analyze_waiting_time_robustness(
    truth_endpoint: Array,
    waiting_times: Sequence[float],
    observable_fn: Callable[[Array, float], Array],
    recover_endpoint: Callable[[float, Array], Any],
    physical_metric_fn: Callable[[Array], Array],
    *,
    jacobian_fn: Callable[[Array, float], Array] | None = None,
    weights: Array | float | None = None,
    rank_rtol: float = 1e-10,
) -> WaitingTimeRobustnessResult:
    """Fit one fixed synthetic endpoint independently at each waiting time.

    ``recover_endpoint(t2, truth_observation)`` is the only optimizer-facing
    interface.  Jacobians are evaluated at each recovered endpoint.  The joint
    information is the sum of the independently measured waiting-time
    information matrices, while its physical normalization is evaluated at the
    fixed truth endpoint.
    """

    truth = np.asarray(truth_endpoint, dtype=float)
    times = tuple(float(value) for value in waiting_times)
    if truth.ndim != 1:
        raise ValueError("truth_endpoint must be one-dimensional")
    if not times or not np.all(np.isfinite(times)):
        raise ValueError("waiting_times must contain finite values")
    if len(set(times)) != len(times):
        raise ValueError("waiting_times must be distinct")

    diagnostics = []
    information_matrices = []
    for waiting_time in times:
        truth_observation = np.asarray(observable_fn(truth, waiting_time))
        endpoint = _endpoint_from_recovery(
            recover_endpoint(waiting_time, truth_observation)
        )
        if endpoint.shape != truth.shape or not np.all(np.isfinite(endpoint)):
            raise ValueError("recovered endpoint has invalid shape or values")
        predicted = np.asarray(observable_fn(endpoint, waiting_time))
        residual = _real_vector(predicted) - _real_vector(truth_observation)
        truth_norm = float(np.linalg.norm(_real_vector(truth_observation)))

        jacobian = (
            _autodiff_jacobian(observable_fn, endpoint, waiting_time)
            if jacobian_fn is None
            else np.asarray(jacobian_fn(endpoint, waiting_time))
        )
        if jacobian.ndim != 2 or jacobian.shape[1] != truth.size:
            raise ValueError("jacobian must have shape (n_residuals, n_controls)")
        information = information_metric(
            jacobian, weights=weights, assume_real_residual=True
        )
        physical_metric = np.asarray(physical_metric_fn(endpoint), dtype=float)
        modes = generalized_observability_modes(
            information, physical_metric, rtol=rank_rtol
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        positive = singular_values[singular_values > rank_rtol * singular_values[0]]
        jacobian_condition = (
            float(singular_values[0] / positive[-1]) if positive.size else np.inf
        )
        residual_norm = float(np.linalg.norm(residual))
        diagnostics.append(
            WaitingTimeDiagnostic(
                waiting_time=waiting_time,
                endpoint=endpoint,
                endpoint_drift=float(np.linalg.norm(endpoint - truth)),
                residual_norm=residual_norm,
                relative_residual_norm=residual_norm / max(truth_norm, np.finfo(float).tiny),
                jacobian_singular_values=singular_values,
                jacobian_condition_number=jacobian_condition,
                generalized_eigenvalues=modes.eigenvalues,
                generalized_effective_rank=modes.effective_rank,
                generalized_condition_number=modes.condition_number,
                information_metric=information,
            )
        )
        information_matrices.append(information)

    joint = joint_information_metric(information_matrices)
    joint_modes = generalized_observability_modes(
        joint, np.asarray(physical_metric_fn(truth), dtype=float), rtol=rank_rtol
    )
    return WaitingTimeRobustnessResult(
        truth_endpoint=truth,
        diagnostics=tuple(diagnostics),
        joint_information_metric=joint,
        joint_generalized_eigenvalues=joint_modes.eigenvalues,
        joint_effective_rank=joint_modes.effective_rank,
        joint_condition_number=joint_modes.condition_number,
    )


__all__ = [
    "WaitingTimeDiagnostic",
    "WaitingTimeRobustnessResult",
    "analyze_waiting_time_robustness",
]
