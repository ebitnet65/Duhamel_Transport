#!/usr/bin/env python3
"""Benchmark the physical route and frame transport to a recovered endpoint.

Endpoint discovery is deliberately upstream of this driver.  The supplied
``q_star`` is held fixed while a physical geodesic is computed from the
reference model and compared with coordinate-space straight interpolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import Liouville_covariant_transport as transport
import Liouville_geodesics as geodesics
from dimer_covariant_geometry import build_dimer_geometry_backend

CONTROL_NAMES = ("J_12", "k_12", "gamma_phi")
DEFAULT_REFERENCE = np.asarray([0.070, 0.003, 0.004], dtype=float)
DEFAULT_BOUNDS = ((-0.10, 0.10), (1.0e-4, 0.03), (1.0e-4, 0.04))


def _array(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite controls")
    return result


def compare_routes_to_endpoint(
    reference: Sequence[float],
    q_star: Sequence[float],
    *,
    n_vertices: int = 9,
    fixed_detuning: float = 0.18,
    beta_12: float = 2.5,
    bounds=DEFAULT_BOUNDS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute geodesic/straight routes and their transported tangent frames."""

    reference_array = _array(reference, "reference")
    endpoint_array = _array(q_star, "q_star")
    started = time.perf_counter()
    geometry = build_dimer_geometry_backend(
        control_mode="coupling_transfer_dephasing",
        fixed_detuning=fixed_detuning,
        beta_12=beta_12,
    )
    geometry_seconds = time.perf_counter() - started

    straight_path = geodesics.linear_path(reference_array, endpoint_array, n_vertices)
    started = time.perf_counter()
    geodesic = geodesics.solve_discrete_geodesic(
        reference_array,
        endpoint_array,
        geometry.metric,
        n_vertices=n_vertices,
        bounds=bounds,
    )
    geodesic_seconds = time.perf_counter() - started
    if not geodesic.success:
        raise RuntimeError(f"geodesic optimization failed: {geodesic.message}")

    initial_frame = transport.metric_orthonormal_frame(geometry.metric(reference_array))
    started = time.perf_counter()
    straight_transport = transport.parallel_transport_frame(
        straight_path,
        initial_frame,
        geometry.christoffel,
        metric_fn=geometry.metric,
    )
    straight_transport_seconds = time.perf_counter() - started
    started = time.perf_counter()
    geodesic_transport = transport.parallel_transport_frame(
        geodesic.path,
        initial_frame,
        geometry.christoffel,
        metric_fn=geometry.metric,
    )
    geodesic_transport_seconds = time.perf_counter() - started
    if not straight_transport.success or not geodesic_transport.success:
        raise RuntimeError("parallel transport failed on at least one route")

    displacement = np.max(np.abs(geodesic.path - straight_path), axis=0)
    endpoint_error = float(np.linalg.norm(geodesic.path[-1] - endpoint_array))
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "model": "exciton_dimer_coupling_transfer_collective_dephasing",
        "control_names": list(CONTROL_NAMES),
        "reference": reference_array.tolist(),
        "q_star": endpoint_array.tolist(),
        "fixed_detuning": float(fixed_detuning),
        "beta_12": float(beta_12),
        "n_vertices": int(n_vertices),
        "geodesic": {
            "success": True,
            "message": geodesic.message,
            "energy": float(geodesic.energy),
            "straight_energy": float(geodesic.initial_energy),
            "relative_energy_reduction": float(geodesic.relative_energy_reduction),
            "iterations": int(geodesic.iterations),
            "function_evaluations": int(geodesic.function_evaluations),
            "endpoint_error": endpoint_error,
            "max_coordinate_displacement": dict(zip(CONTROL_NAMES, displacement.tolist())),
            "minimum_metric_determinant": float(np.min(geodesic.metric_determinants)),
            "maximum_metric_condition": float(np.max(geodesic.metric_condition_numbers)),
        },
        "transport": {
            "straight_success": True,
            "geodesic_success": True,
            "straight_rhs_evaluations": int(straight_transport.nfev),
            "geodesic_rhs_evaluations": int(geodesic_transport.nfev),
            "straight_max_gram_error": float(
                np.max(straight_transport.metric_gram_error)
            ),
            "geodesic_max_gram_error": float(
                np.max(geodesic_transport.metric_gram_error)
            ),
            "endpoint_frame_frobenius_difference": float(
                np.linalg.norm(
                    geodesic_transport.final_frame - straight_transport.final_frame,
                    ord="fro",
                )
            ),
        },
        "timing_seconds": {
            "geometry_build": geometry_seconds,
            "geodesic_solve": geodesic_seconds,
            "straight_transport": straight_transport_seconds,
            "geodesic_transport": geodesic_transport_seconds,
        },
    }
    arrays = {
        "reference": reference_array,
        "q_star": endpoint_array,
        "straight_path": straight_path,
        "geodesic_path": geodesic.path,
        "initial_frame": initial_frame,
        "straight_frames": straight_transport.frames,
        "geodesic_frames": geodesic_transport.frames,
        "metric_determinants": geodesic.metric_determinants,
        "metric_condition_numbers": geodesic.metric_condition_numbers,
    }
    return diagnostics, arrays


def write_results(
    diagnostics: dict[str, Any], arrays: dict[str, np.ndarray], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transport_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "transport_paths.npz", **arrays)


def _endpoint_from_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("q_star", "recovered_endpoint", "endpoint"):
        if key in payload:
            return _array(payload[key], key)
    recovery = payload.get("recovery")
    if isinstance(recovery, dict) and "endpoint" in recovery:
        return _array(recovery["endpoint"], "recovery.endpoint")
    raise ValueError(
        f"{path} must contain q_star, recovered_endpoint, endpoint, "
        "or recovery.endpoint"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    endpoint_group = parser.add_mutually_exclusive_group(required=True)
    endpoint_group.add_argument("--endpoint", type=float, nargs=3, metavar=("J", "K", "GAMMA"))
    endpoint_group.add_argument("--endpoint-json", type=Path)
    parser.add_argument("--reference", type=float, nargs=3, default=DEFAULT_REFERENCE)
    parser.add_argument("--vertices", type=int, default=9)
    parser.add_argument("--fixed-detuning", type=float, default=0.18)
    parser.add_argument("--beta-12", type=float, default=2.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    endpoint = (
        _endpoint_from_json(args.endpoint_json)
        if args.endpoint_json is not None
        else args.endpoint
    )
    diagnostics, arrays = compare_routes_to_endpoint(
        args.reference,
        endpoint,
        n_vertices=args.vertices,
        fixed_detuning=args.fixed_detuning,
        beta_12=args.beta_12,
    )
    write_results(diagnostics, arrays, args.output_dir)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
