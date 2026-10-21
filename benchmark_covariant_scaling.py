#!/usr/bin/env python3
"""Benchmark covariant geometry, geodesics, and transport versus dimension."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import Liouville_covariant_transport as transport
import Liouville_geodesics as geodesics


@dataclass(frozen=True)
class ScalingRecord:
    dimension: int
    n_vertices: int
    compile_seconds: float
    metric_eval_ms: float
    christoffel_eval_ms: float
    geodesic_seconds: float
    geodesic_iterations: int
    geodesic_evaluations: int
    straight_energy: float
    geodesic_energy: float
    relative_energy_reduction: float
    transport_seconds: float
    transport_rhs_evaluations: int
    max_metric_condition: float
    max_gram_error: float


def build_synthetic_backend(dimension: int, coupling: float = 0.18):
    """Return compiled metric and connection for a curved embedding family."""

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError("JAX is required for the scaling benchmark.") from exc
    jax.config.update("jax_enable_x64", True)

    def coordinates(lam):
        root = jnp.sqrt(lam)
        neighbor = coupling * jnp.sqrt(lam * jnp.roll(lam, -1))
        collective = (coupling / np.sqrt(dimension)) * jnp.asarray(
            [jnp.sum(jnp.log(lam)), jnp.sum(root)]
        )
        return jnp.concatenate((root, neighbor, collective))

    def metric_fn(lam):
        jac = jax.jacfwd(coordinates)(lam)
        return jac.T @ jac

    def christoffel_fn(lam):
        metric = metric_fn(lam)
        dmetric = jnp.moveaxis(jax.jacfwd(metric_fn)(lam), -1, 0)
        covariant = 0.5 * (
            jnp.transpose(dmetric, (1, 0, 2))
            + jnp.transpose(dmetric, (1, 2, 0))
            - dmetric
        )
        return jnp.einsum("ab,bmn->amn", jnp.linalg.inv(metric), covariant)

    metric_jit = jax.jit(metric_fn)
    christoffel_jit = jax.jit(christoffel_fn)
    reference = jnp.linspace(0.2, 0.8, dimension)
    start = time.perf_counter()
    metric_jit(reference).block_until_ready()
    christoffel_jit(reference).block_until_ready()
    compile_seconds = time.perf_counter() - start

    def metric(lam):
        return np.asarray(metric_jit(jnp.asarray(lam, dtype=float)))

    def christoffel(lam):
        return np.asarray(christoffel_jit(jnp.asarray(lam, dtype=float)))

    return metric, christoffel, compile_seconds


def _average_time(fn, argument, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn(argument)
    return (time.perf_counter() - start) / repeats


def benchmark_dimension(
    dimension: int,
    *,
    n_vertices: int = 7,
    repeats: int = 20,
) -> ScalingRecord:
    metric, christoffel, compile_seconds = build_synthetic_backend(dimension)
    start_point = np.linspace(0.18, 0.72, dimension)
    truth_point = np.linspace(0.82, 0.28, dimension)

    metric_eval = _average_time(metric, start_point, repeats)
    christoffel_eval = _average_time(christoffel, start_point, repeats)

    start = time.perf_counter()
    geodesic = geodesics.solve_discrete_geodesic(
        start_point,
        truth_point,
        metric,
        n_vertices=n_vertices,
        bounds=[(0.08, 0.95)] * dimension,
        options={"maxiter": 400},
    )
    geodesic_seconds = time.perf_counter() - start
    if not geodesic.success:
        raise RuntimeError(
            f"dimension {dimension} geodesic failed: {geodesic.message}"
        )

    frame = transport.metric_orthonormal_frame(metric(start_point))
    start = time.perf_counter()
    transported = transport.parallel_transport_frame(
        geodesic.path,
        frame,
        christoffel,
        metric_fn=metric,
        rtol=1e-8,
        atol=1e-10,
    )
    transport_seconds = time.perf_counter() - start
    if not transported.success:
        raise RuntimeError(
            f"dimension {dimension} transport failed: {transported.message}"
        )

    return ScalingRecord(
        dimension=dimension,
        n_vertices=n_vertices,
        compile_seconds=compile_seconds,
        metric_eval_ms=1e3 * metric_eval,
        christoffel_eval_ms=1e3 * christoffel_eval,
        geodesic_seconds=geodesic_seconds,
        geodesic_iterations=geodesic.iterations,
        geodesic_evaluations=geodesic.function_evaluations,
        straight_energy=geodesic.initial_energy,
        geodesic_energy=geodesic.energy,
        relative_energy_reduction=geodesic.relative_energy_reduction,
        transport_seconds=transport_seconds,
        transport_rhs_evaluations=transported.nfev,
        max_metric_condition=float(np.max(geodesic.metric_condition_numbers)),
        max_gram_error=float(np.max(transported.metric_gram_error)),
    )


def run_scaling_benchmark(
    dimensions=(2, 3, 4, 6, 8), *, n_vertices=7, repeats=20
):
    return [
        benchmark_dimension(dim, n_vertices=n_vertices, repeats=repeats)
        for dim in dimensions
    ]


def write_results(records: list[ScalingRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    (output_dir / "covariant_scaling.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (output_dir / "covariant_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    parser.add_argument("--vertices", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("covariant_scaling_benchmark"))
    args = parser.parse_args()
    records = run_scaling_benchmark(
        args.dimensions, n_vertices=args.vertices, repeats=args.repeats
    )
    write_results(records, args.output_dir)
    print(json.dumps([asdict(record) for record in records], indent=2))


if __name__ == "__main__":
    main()
