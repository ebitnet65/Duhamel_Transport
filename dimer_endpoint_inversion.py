"""Bounded endpoint recovery from a fixed dimer 2D spectral observation.

This module deliberately contains no path or geodesic dependence.  It solves

    q* = argmin_q 1/2 ||O(q; t2) - y_obs||_2^2 / ||y_obs||_2^2

for ``q = (J_12, k_12, gamma_phi)`` using exact JAX gradients and deterministic
multistart L-BFGS-B.  Route discovery is a separate downstream calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

ArrayLike = Any
PackedObservable = Callable[[ArrayLike, float], ArrayLike]


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError("JAX is required for dimer endpoint inversion.") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


@dataclass(frozen=True)
class EndpointRecoveryConfig:
    """Numerical settings for deterministic bounded multistart recovery."""

    bounds: tuple[tuple[float, float], ...] = (
        (-0.10, 0.10),
        (1.0e-4, 0.03),
        (1.0e-4, 0.04),
    )
    starts: tuple[tuple[float, ...], ...] = ()
    n_random_starts: int = 4
    seed: int = 0
    maxiter: int = 500
    ftol: float = 1.0e-14
    gtol: float = 1.0e-9


@dataclass(frozen=True)
class SyntheticTruth:
    endpoint: np.ndarray
    t2: float
    noiseless_observation: np.ndarray
    observation: np.ndarray
    relative_noise: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint.tolist(),
            "t2": self.t2,
            "noiseless_observation": self.noiseless_observation.tolist(),
            "observation": self.observation.tolist(),
            "relative_noise": self.relative_noise,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RecoveryRun:
    start: np.ndarray
    endpoint: np.ndarray
    success: bool
    message: str
    objective: float
    gradient_norm: float
    iterations: int
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = self.start.tolist()
        data["endpoint"] = self.endpoint.tolist()
        return data


@dataclass(frozen=True)
class EndpointRecoveryResult:
    endpoint: np.ndarray
    best_start: np.ndarray
    success: bool
    message: str
    objective: float
    relative_spectral_residual: float
    endpoint_absolute_error: float | None
    endpoint_relative_error: float | None
    gradient_norm: float
    iterations: int
    evaluations: int
    jacobian_singular_values: np.ndarray
    jacobian_condition_number: float | None
    jacobian_rank: int
    runs: tuple[RecoveryRun, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint.tolist(),
            "best_start": self.best_start.tolist(),
            "success": self.success,
            "message": self.message,
            "objective": self.objective,
            "relative_spectral_residual": self.relative_spectral_residual,
            "endpoint_absolute_error": self.endpoint_absolute_error,
            "endpoint_relative_error": self.endpoint_relative_error,
            "gradient_norm": self.gradient_norm,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "jacobian_singular_values": self.jacobian_singular_values.tolist(),
            "jacobian_condition_number": self.jacobian_condition_number,
            "jacobian_rank": self.jacobian_rank,
            "runs": [run.to_dict() for run in self.runs],
        }


def generate_synthetic_truth(
    observable: PackedObservable,
    q_true: Sequence[float],
    t2: float,
    *,
    relative_noise: float = 0.0,
    seed: int = 0,
) -> SyntheticTruth:
    """Generate a path-independent packed spectral observation.

    ``relative_noise`` is the expected Euclidean noise norm divided by the
    noiseless signal norm.  Noise is deterministic for a fixed seed.
    """

    if relative_noise < 0.0:
        raise ValueError("relative_noise must be nonnegative")
    endpoint = np.asarray(q_true, dtype=float).reshape(-1)
    noiseless = np.asarray(observable(endpoint, float(t2)), dtype=float).reshape(-1)
    if not np.all(np.isfinite(noiseless)):
        raise ValueError("observable returned non-finite values")
    rng = np.random.default_rng(seed)
    noise_scale = relative_noise * np.linalg.norm(noiseless) / np.sqrt(noiseless.size)
    observation = noiseless + noise_scale * rng.standard_normal(noiseless.shape)
    return SyntheticTruth(
        endpoint=endpoint,
        t2=float(t2),
        noiseless_observation=noiseless,
        observation=observation,
        relative_noise=float(relative_noise),
        seed=int(seed),
    )


def _multistarts(config: EndpointRecoveryConfig) -> tuple[np.ndarray, ...]:
    bounds = np.asarray(config.bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must contain (lower, upper) pairs")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be less than its upper bound")
    if config.n_random_starts < 0:
        raise ValueError("n_random_starts must be nonnegative")
    starts = [0.5 * (bounds[:, 0] + bounds[:, 1])]
    starts.extend(np.asarray(start, dtype=float).reshape(-1) for start in config.starts)
    rng = np.random.default_rng(config.seed)
    starts.extend(
        rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(config.n_random_starts)
    )
    unique: list[np.ndarray] = []
    for start in starts:
        if start.size != bounds.shape[0]:
            raise ValueError("every start must match the number of bounds")
        if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
            raise ValueError("all starts must lie inside bounds")
        if not any(np.array_equal(start, existing) for existing in unique):
            unique.append(start)
    return tuple(unique)


def recover_dimer_endpoint(
    observable: PackedObservable,
    observed: Sequence[float],
    t2: float,
    *,
    config: EndpointRecoveryConfig = EndpointRecoveryConfig(),
    truth_endpoint: Sequence[float] | None = None,
) -> EndpointRecoveryResult:
    """Recover the fixed endpoint from one packed 2D spectrum."""

    jax, jnp = _require_jax()
    observed_np = np.asarray(observed, dtype=float).reshape(-1)
    if observed_np.size == 0 or not np.all(np.isfinite(observed_np)):
        raise ValueError("observed must be a nonempty finite vector")
    if len(config.bounds) != 3:
        raise ValueError("the dimer endpoint requires exactly three bounds")
    scale = max(float(np.linalg.norm(observed_np)), np.finfo(float).tiny)
    observed_jax = jnp.asarray(observed_np, dtype=jnp.float64)

    def loss(q):
        residual = (observable(q, t2) - observed_jax) / scale
        return 0.5 * jnp.vdot(residual, residual)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    # Compile outside SciPy so compilation is not attributed to an evaluation.
    warm_start = jnp.asarray(_multistarts(config)[0], dtype=jnp.float64)
    _warm_value, warm_gradient = value_and_grad(warm_start)
    warm_gradient.block_until_ready()

    def scipy_objective(q):
        value, gradient = value_and_grad(jnp.asarray(q, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=float)

    runs = []
    for start in _multistarts(config):
        result = minimize(
            scipy_objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=config.bounds,
            options={
                "maxiter": config.maxiter,
                "ftol": config.ftol,
                "gtol": config.gtol,
            },
        )
        runs.append(
            RecoveryRun(
                start=np.asarray(start, dtype=float),
                endpoint=np.asarray(result.x, dtype=float),
                success=bool(result.success),
                message=str(result.message),
                objective=float(result.fun),
                gradient_norm=float(np.linalg.norm(result.jac)),
                iterations=int(result.nit),
                evaluations=int(result.nfev),
            )
        )
    converged = [run for run in runs if run.success and np.all(np.isfinite(run.endpoint))]
    best = min(converged or runs, key=lambda run: run.objective)
    predicted = np.asarray(observable(best.endpoint, t2), dtype=float).reshape(-1)
    relative_residual = float(np.linalg.norm(predicted - observed_np) / scale)
    jacobian = np.asarray(
        jax.jacfwd(lambda q: observable(q, t2))(
            jnp.asarray(best.endpoint, dtype=jnp.float64)
        ),
        dtype=float,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = (
        None
        if singular_values[-1] == 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    rank = int(np.linalg.matrix_rank(jacobian))
    absolute_error = None
    relative_error = None
    if truth_endpoint is not None:
        truth = np.asarray(truth_endpoint, dtype=float).reshape(-1)
        if truth.size != best.endpoint.size:
            raise ValueError("truth_endpoint has the wrong dimension")
        absolute_error = float(np.linalg.norm(best.endpoint - truth))
        relative_error = float(absolute_error / max(np.linalg.norm(truth), np.finfo(float).tiny))
    successful = bool(best.success and np.all(np.isfinite(best.endpoint)))
    return EndpointRecoveryResult(
        endpoint=best.endpoint,
        best_start=best.start,
        success=successful,
        message=best.message,
        objective=best.objective,
        relative_spectral_residual=relative_residual,
        endpoint_absolute_error=absolute_error,
        endpoint_relative_error=relative_error,
        gradient_norm=best.gradient_norm,
        iterations=best.iterations,
        evaluations=best.evaluations,
        jacobian_singular_values=singular_values,
        jacobian_condition_number=condition,
        jacobian_rank=rank,
        runs=tuple(runs),
    )


__all__ = [
    "EndpointRecoveryConfig",
    "EndpointRecoveryResult",
    "RecoveryRun",
    "SyntheticTruth",
    "generate_synthetic_truth",
    "recover_dimer_endpoint",
]
