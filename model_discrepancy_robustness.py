"""Endpoint robustness and uncertainty windows under model discrepancy.

The distorted-truth generator and ideal-model inversion are injected.  This
keeps artifact physics and optimization in their owning modules while this
layer performs tangent-space residual decomposition and reliability analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spectral_observability import generalized_observability_modes, information_metric

Array = np.ndarray


@dataclass(frozen=True)
class ReliabilityThresholds:
    """Empirical acceptance thresholds; these are not mathematical guarantees."""

    maximum_relative_residual: float = 1.0e-2
    maximum_relative_orthogonal_residual: float = 1.0e-2
    maximum_orthogonal_residual_fraction: float = 1.0
    maximum_euclidean_endpoint_bias: float = 5.0e-3
    maximum_physical_endpoint_bias: float = 0.25
    maximum_jacobian_condition_number: float = 1.0e6
    require_full_rank: bool = True


@dataclass(frozen=True)
class DiscrepancyDiagnostic:
    artifact_strength: float
    replicate_seed: int | None
    endpoint: Array
    optimizer_success: bool
    endpoint_bias: float
    physical_endpoint_bias: float
    residual_norm: float
    relative_residual: float
    tangent_parallel_residual_norm: float
    tangent_orthogonal_residual_norm: float
    relative_orthogonal_residual: float
    orthogonal_residual_fraction: float
    jacobian_singular_values: Array
    jacobian_rank: int
    jacobian_condition_number: float
    information_metric: Array
    generalized_eigenvalues: Array
    generalized_effective_rank: int
    local_covariance: Array
    physical_uncertainty_rms: float
    downstream_changes: Mapping[str, float]
    reliable: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DiscrepancySweepResult:
    truth_endpoint: Array
    artifact_strengths: Array
    replicate_seeds: tuple[int | None, ...]
    thresholds: ReliabilityThresholds
    diagnostics: tuple[DiscrepancyDiagnostic, ...]
    uncertainty_window: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return convert(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                if np.isnan(value):
                    return "NaN"
                return "Infinity" if value > 0.0 else "-Infinity"
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def decompose_residual(
    residual: Array,
    jacobian: Array,
    *,
    rcond: float = 1.0e-10,
) -> tuple[Array, Array, Array, int]:
    """Project a residual parallel and orthogonal to the Jacobian tangent space."""

    vector = np.asarray(residual, dtype=float).reshape(-1)
    derivative = np.asarray(jacobian, dtype=float)
    if derivative.ndim != 2 or derivative.shape[0] != vector.size:
        raise ValueError("jacobian rows must match the residual size")
    left, singular_values, _right_h = np.linalg.svd(derivative, full_matrices=False)
    scale = singular_values[0] if singular_values.size else 0.0
    rank = int(np.count_nonzero(singular_values > rcond * scale))
    tangent = left[:, :rank]
    parallel = tangent @ (tangent.T @ vector) if rank else np.zeros_like(vector)
    orthogonal = vector - parallel
    return parallel, orthogonal, singular_values, rank


def _endpoint_from_recovery(result: Any) -> tuple[Array, bool]:
    success = bool(getattr(result, "success", True))
    if isinstance(result, Mapping):
        success = bool(result.get("success", success))
        for name in ("endpoint", "parameters", "x"):
            if name in result:
                return np.asarray(result[name], dtype=float), success
    if isinstance(result, (np.ndarray, list, tuple)):
        return np.asarray(result, dtype=float), success
    for name in ("endpoint", "parameters", "x"):
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float), success
    raise TypeError("recovery result must expose endpoint, parameters, or x")


def _flatten_diagnostics(values: Mapping[str, Any]) -> dict[str, Array]:
    return {name: np.asarray(value) for name, value in values.items()}


def analyze_model_discrepancy_sweep(
    truth_endpoint: Array,
    artifact_strengths: Sequence[float],
    truth_generator: Callable[[float, int | None], Array],
    ideal_observable: Callable[[Array], Array],
    recover_endpoint: Callable[[Array], Any],
    jacobian_fn: Callable[[Array], Array],
    physical_metric_fn: Callable[[Array], Array],
    *,
    replicate_seeds: Sequence[int | None] = (None,),
    thresholds: ReliabilityThresholds = ReliabilityThresholds(),
    rank_rtol: float = 1.0e-10,
    downstream_fn: Callable[[Array], Mapping[str, Any]] | None = None,
) -> DiscrepancySweepResult:
    """Recover endpoints across a deterministic artifact-strength grid.

    ``truth_generator(strength, seed)`` makes later noise ensembles explicit;
    use the default single ``None`` seed for a noise-free deterministic sweep.
    Local covariance uses the orthogonal-residual RMS as a discrepancy scale,
    not as a claim of statistical posterior uncertainty.
    """

    truth = np.asarray(truth_endpoint, dtype=float).reshape(-1)
    strengths = np.asarray(artifact_strengths, dtype=float)
    seeds = tuple(replicate_seeds)
    if strengths.ndim != 1 or strengths.size == 0 or not np.all(np.isfinite(strengths)):
        raise ValueError("artifact_strengths must be a nonempty finite vector")
    if np.any(np.diff(strengths) <= 0.0):
        raise ValueError("artifact_strengths must be strictly increasing")
    if not np.isclose(strengths[0], 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("artifact_strengths must start at zero")
    if not seeds:
        raise ValueError("at least one replicate seed is required")

    reference_downstream = (
        _flatten_diagnostics(downstream_fn(truth)) if downstream_fn is not None else {}
    )
    diagnostics = []
    for strength in strengths:
        for seed in seeds:
            observed = np.asarray(truth_generator(float(strength), seed))
            endpoint, optimizer_success = _endpoint_from_recovery(
                recover_endpoint(observed)
            )
            endpoint = endpoint.reshape(-1)
            if endpoint.shape != truth.shape or not np.all(np.isfinite(endpoint)):
                raise ValueError("recovered endpoint has invalid shape or values")
            prediction = np.asarray(ideal_observable(endpoint))
            if prediction.shape != observed.shape:
                raise ValueError("truth and ideal observable shapes differ")
            residual = (prediction - observed).reshape(-1)
            if np.iscomplexobj(residual):
                residual = np.concatenate((residual.real, residual.imag))
            residual = np.asarray(residual, dtype=float)
            jacobian = np.asarray(jacobian_fn(endpoint))
            if np.iscomplexobj(jacobian):
                flat = jacobian.reshape((-1, endpoint.size))
                jacobian = np.vstack((flat.real, flat.imag))
            else:
                jacobian = jacobian.reshape((-1, endpoint.size)).astype(float)
            parallel, orthogonal, singular_values, rank = decompose_residual(
                residual, jacobian, rcond=rank_rtol
            )
            info = information_metric(jacobian, assume_real_residual=True)
            metric = np.asarray(physical_metric_fn(endpoint), dtype=float)
            modes = generalized_observability_modes(info, metric, rtol=rank_rtol)
            endpoint_delta = endpoint - truth
            truth_metric = np.asarray(physical_metric_fn(truth), dtype=float)
            physical_bias = float(np.sqrt(max(endpoint_delta @ truth_metric @ endpoint_delta, 0.0)))
            residual_norm = float(np.linalg.norm(residual))
            observed_norm = float(np.linalg.norm(np.asarray(observed).reshape(-1)))
            orthogonal_norm = float(np.linalg.norm(orthogonal))
            degrees = max(residual.size - rank, 1)
            discrepancy_variance = orthogonal_norm**2 / degrees
            covariance = discrepancy_variance * np.linalg.pinv(info, rcond=rank_rtol)
            physical_uncertainty = float(
                np.sqrt(max(np.trace(metric @ covariance), 0.0))
            )
            positive = singular_values[singular_values > rank_rtol * singular_values[0]]
            condition = float(singular_values[0] / positive[-1]) if positive.size else np.inf

            downstream_changes = {}
            if downstream_fn is not None:
                current = _flatten_diagnostics(downstream_fn(endpoint))
                if current.keys() != reference_downstream.keys():
                    raise ValueError("downstream diagnostics changed keys")
                downstream_changes = {
                    name: float(np.linalg.norm(current[name] - reference_downstream[name]))
                    for name in current
                }

            relative_residual = residual_norm / max(observed_norm, np.finfo(float).tiny)
            residual_floor = np.finfo(float).eps * max(observed_norm, 1.0)
            orthogonal_fraction = (
                0.0 if residual_norm <= residual_floor else orthogonal_norm / residual_norm
            )
            relative_orthogonal = orthogonal_norm / max(
                observed_norm, np.finfo(float).tiny
            )
            endpoint_bias = float(np.linalg.norm(endpoint_delta))
            reasons = []
            if not optimizer_success:
                reasons.append("optimizer_failure")
            if relative_residual > thresholds.maximum_relative_residual:
                reasons.append("relative_residual")
            if relative_orthogonal > thresholds.maximum_relative_orthogonal_residual:
                reasons.append("relative_orthogonal_residual")
            if orthogonal_fraction > thresholds.maximum_orthogonal_residual_fraction:
                reasons.append("orthogonal_residual_fraction")
            if endpoint_bias > thresholds.maximum_euclidean_endpoint_bias:
                reasons.append("endpoint_bias")
            if physical_bias > thresholds.maximum_physical_endpoint_bias:
                reasons.append("physical_endpoint_bias")
            if condition > thresholds.maximum_jacobian_condition_number:
                reasons.append("jacobian_condition")
            if thresholds.require_full_rank and rank < truth.size:
                reasons.append("jacobian_rank")
            diagnostics.append(
                DiscrepancyDiagnostic(
                    artifact_strength=float(strength),
                    replicate_seed=seed,
                    endpoint=endpoint,
                    optimizer_success=optimizer_success,
                    endpoint_bias=endpoint_bias,
                    physical_endpoint_bias=physical_bias,
                    residual_norm=residual_norm,
                    relative_residual=relative_residual,
                    tangent_parallel_residual_norm=float(np.linalg.norm(parallel)),
                    tangent_orthogonal_residual_norm=orthogonal_norm,
                    relative_orthogonal_residual=relative_orthogonal,
                    orthogonal_residual_fraction=orthogonal_fraction,
                    jacobian_singular_values=singular_values,
                    jacobian_rank=rank,
                    jacobian_condition_number=condition,
                    information_metric=info,
                    generalized_eigenvalues=modes.eigenvalues,
                    generalized_effective_rank=modes.effective_rank,
                    local_covariance=covariance,
                    physical_uncertainty_rms=physical_uncertainty,
                    downstream_changes=downstream_changes,
                    reliable=not reasons,
                    failure_reasons=tuple(reasons),
                )
            )

    reliable_strengths = []
    for strength in strengths:
        at_strength = [item for item in diagnostics if item.artifact_strength == strength]
        if all(item.reliable for item in at_strength):
            reliable_strengths.append(float(strength))
        else:
            break
    uncertainty_window = (
        (float(strengths[0]), reliable_strengths[-1]) if reliable_strengths else None
    )
    return DiscrepancySweepResult(
        truth_endpoint=truth,
        artifact_strengths=strengths,
        replicate_seeds=seeds,
        thresholds=thresholds,
        diagnostics=tuple(diagnostics),
        uncertainty_window=uncertainty_window,
    )


__all__ = [
    "DiscrepancyDiagnostic",
    "DiscrepancySweepResult",
    "ReliabilityThresholds",
    "analyze_model_discrepancy_sweep",
    "decompose_residual",
]
