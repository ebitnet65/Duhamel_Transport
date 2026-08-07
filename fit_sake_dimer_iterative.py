#!/usr/bin/env python3
"""Iterative trust-region SAKE recovery of dimer control parameters.

Synthetic target pathway states are generated once by direct inversion. Each
iteration constructs a local third-order SAKE chart, minimizes its pathway-state
residual inside a trust region, and performs one direct calculation to accept or
reject the proposed step. Accepted exact overlap maps are composed to retain the
total mixing of the initial pathways into the recovered target pathways.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
import scipy.optimize

import Liouville_transport_general as general
from benchmark_sake_accuracy_timing import (
    SectorData,
    block_diag,
    build_sector_data,
    clean_dimer_parameter_map,
    load_derivative_cache,
    validate_derivative_cache,
)
from exciton_dimer_model import exciton_dimer_model
from fit_v_system_transport_signal import pathway_state_grid_matrix, project_transport_derivative


@dataclass(frozen=True)
class Chart:
    parameter_map: general.ParameterMap
    sectors: list[SectorData]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega0", type=float, default=2.0)
    parser.add_argument("--delta-omega", type=float, default=0.18)
    parser.add_argument("--delta-J", type=float, default=None)
    parser.add_argument("--delta-k", type=float, default=None)
    parser.add_argument("--reference-J", type=float, default=0.0)
    parser.add_argument("--reference-k", type=float, default=0.0)
    parser.add_argument("--target-J", type=float, default=-0.1)
    parser.add_argument("--target-k", type=float, default=0.012)
    parser.add_argument("--J-min", type=float, default=None)
    parser.add_argument("--J-max", type=float, default=None)
    parser.add_argument("--k-min", type=float, default=0.0)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--initial-radius", type=float, default=0.005)
    parser.add_argument("--minimum-radius", type=float, default=2.5e-4)
    parser.add_argument("--maximum-radius", type=float, default=0.0125)
    parser.add_argument("--parameter-tolerance", type=float, default=5e-4)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--acceptance-threshold", type=float, default=0.1)
    parser.add_argument("--n-omega", type=int, default=8)
    parser.add_argument("--omega-min", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3e-3)
    parser.add_argument("--n-quad", type=int, default=4)
    parser.add_argument("--rcond", type=float, default=1e-11)
    parser.add_argument("--basis-rtol", type=float, default=1e-9)
    parser.add_argument("--clean-gamma", type=float, default=0.01)
    parser.add_argument("--clean-gamma-phi", type=float, default=0.01)
    parser.add_argument("--clean-mu-esa-scale", type=float, default=1.0)
    parser.add_argument("--sectors", nargs="+", choices=("RP", "NRP", "DQC"), default=("RP", "NRP"))
    parser.add_argument("--derivative-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dimer_backend_benchmarks/iterative_inverse"),
    )
    return parser.parse_args()


def args_at_reference(args: argparse.Namespace, reference: np.ndarray) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.reference_J = float(reference[0])
    copied.reference_k = float(reference[1])
    return copied


def affine_derivatives_at_reference(
    template: dict[tuple[int, ...], np.ndarray],
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
) -> dict[tuple[int, ...], np.ndarray]:
    derivatives = {key: np.array(value, copy=True) for key, value in template.items()}
    derivatives[()] = model.L(parameter_map.params(parameter_map.zero()))
    return derivatives


def direct_states(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    pathways_by_sector: list[tuple[str, list[general.CanonicalPathway]]],
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    return [
        pathway_state_grid_matrix(
            model,
            parameter_map,
            parameter_map.zero(),
            omega1_values,
            omega3_values,
            pathways,
            tau2=args.tau2,
            eta=args.eta,
        )
        for _, pathways in pathways_by_sector
    ]


def normalized_squared_error(actual: list[np.ndarray], predicted: list[np.ndarray]) -> float:
    numerator = sum(float(np.linalg.norm(a - p) ** 2) for a, p in zip(actual, predicted, strict=True))
    denominator = sum(float(np.linalg.norm(a) ** 2) for a in actual)
    return numerator / max(denominator, 1e-28)


def surrogate_states(chart: Chart, displacement: np.ndarray) -> list[np.ndarray]:
    return [
        data.reference_pathway_states @ data.expansion.evaluate(displacement, order=3).T
        for data in chart.sectors
    ]


def exact_step_transport(
    reference_states: list[np.ndarray],
    trial_states: list[np.ndarray],
    rcond: float,
) -> np.ndarray:
    return block_diag([
        project_transport_derivative(reference, trial, rcond=rcond)[0]
        for reference, trial in zip(reference_states, trial_states, strict=True)
    ])


def write_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = exciton_dimer_model()
    omega_min = args.omega0 - 0.3 if args.omega_min is None else args.omega_min
    omega_max = args.omega0 + 0.3 if args.omega_max is None else args.omega_max
    omega1_values = np.linspace(omega_min, omega_max, args.n_omega)
    omega3_values = np.linspace(omega_min, omega_max, args.n_omega)
    sectors = tuple(sector.upper() for sector in args.sectors)

    initial_reference = np.array([args.reference_J, args.reference_k], dtype=float)
    target = np.array([args.target_J, args.target_k], dtype=float)
    initial_args = args_at_reference(args, initial_reference)
    initial_map = clean_dimer_parameter_map(initial_args)
    derivative_template, derivative_seconds, metadata = load_derivative_cache(args.derivative_cache)
    validate_derivative_cache(metadata, initial_map, initial_args)
    initial_derivatives = affine_derivatives_at_reference(derivative_template, model, initial_map)
    initial_chart_start = time.perf_counter()
    initial_sector_data, _, _ = build_sector_data(
        model, initial_map, omega1_values, omega3_values, sectors, initial_derivatives, args
    )
    initial_chart_seconds = time.perf_counter() - initial_chart_start
    pathways_by_sector = [(data.sector, data.pathways) for data in initial_sector_data]

    target_args = args_at_reference(args, target)
    target_map = clean_dimer_parameter_map(target_args)
    target_direct_start = time.perf_counter()
    target_states = direct_states(
        model, target_map, pathways_by_sector, omega1_values, omega3_values, args
    )
    target_direct_seconds = time.perf_counter() - target_direct_start

    reference = initial_reference.copy()
    chart = Chart(initial_map, initial_sector_data)
    reference_states = [data.reference_pathway_states for data in chart.sectors]
    initial_reference_states = [np.array(states, copy=True) for states in reference_states]
    current_error = normalized_squared_error(target_states, reference_states)
    total_dimension = sum(len(data.pathways) for data in chart.sectors)
    composed_local_transport = np.eye(total_dimension, dtype=complex)
    radius = args.initial_radius
    rows: list[dict[str, float | int | bool]] = []
    direct_validations = 0
    iterative_start = time.perf_counter()

    for iteration in range(args.max_iterations):
        iteration_start = time.perf_counter()
        scale = np.array([control.scale for control in chart.parameter_map.controls])
        target_displacement = (target - reference) / scale
        if np.linalg.norm(target_displacement) <= args.parameter_tolerance:
            break

        def objective(displacement: np.ndarray) -> float:
            return normalized_squared_error(target_states, surrogate_states(chart, displacement))

        constraint = {"type": "ineq", "fun": lambda x: radius**2 - float(np.dot(x, x))}
        physical_lower = np.array([
            -np.inf if args.J_min is None else args.J_min,
            -np.inf if args.k_min is None else args.k_min,
        ])
        physical_upper = np.array([
            np.inf if args.J_max is None else args.J_max,
            np.inf if args.k_max is None else args.k_max,
        ])
        displacement_bounds = scipy.optimize.Bounds(
            (physical_lower - reference) / scale,
            (physical_upper - reference) / scale,
        )
        x0 = target_displacement.copy()
        x0_norm = np.linalg.norm(x0)
        if x0_norm > 0.8 * radius:
            x0 *= 0.8 * radius / x0_norm
        optimizer_start = time.perf_counter()
        result = scipy.optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=displacement_bounds,
            constraints=(constraint,),
            options={"ftol": 1e-12, "maxiter": 200},
        )
        optimizer_seconds = time.perf_counter() - optimizer_start
        step = np.asarray(result.x, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if step_norm > radius * (1.0 + 1e-6):
            step *= radius / step_norm
            step_norm = radius
        trial_reference = reference + scale * step
        trial_args = args_at_reference(args, trial_reference)
        trial_map = clean_dimer_parameter_map(trial_args)
        validation_start = time.perf_counter()
        trial_states = direct_states(
            model, trial_map, pathways_by_sector, omega1_values, omega3_values, args
        )
        direct_validation_seconds = time.perf_counter() - validation_start
        direct_validations += 1
        trial_error = normalized_squared_error(target_states, trial_states)
        predicted_trial_error = objective(step)
        predicted_reduction = max(current_error - predicted_trial_error, 1e-16)
        actual_reduction = current_error - trial_error
        rho = actual_reduction / predicted_reduction
        accepted = bool(actual_reduction > 0.0 and rho >= args.acceptance_threshold)

        row = {
            "iteration": iteration,
            "reference_J": float(reference[0]),
            "reference_k": float(reference[1]),
            "trial_J": float(trial_reference[0]),
            "trial_k": float(trial_reference[1]),
            "radius": radius,
            "step_norm": step_norm,
            "current_state_error": float(np.sqrt(current_error)),
            "predicted_trial_error": float(np.sqrt(max(predicted_trial_error, 0.0))),
            "direct_trial_error": float(np.sqrt(trial_error)),
            "rho": float(rho),
            "accepted": accepted,
            "optimizer_seconds": optimizer_seconds,
            "direct_validation_seconds": direct_validation_seconds,
            "transport_projection_seconds": 0.0,
            "chart_rebuild_seconds": 0.0,
            "iteration_total_seconds": 0.0,
            "cumulative_iterative_seconds": 0.0,
        }

        if not accepted:
            radius *= 0.5
            row["iteration_total_seconds"] = time.perf_counter() - iteration_start
            row["cumulative_iterative_seconds"] = time.perf_counter() - iterative_start
            rows.append(row)
            if radius < args.minimum_radius:
                break
            continue

        projection_start = time.perf_counter()
        step_transport = exact_step_transport(reference_states, trial_states, args.rcond)
        composed_local_transport = step_transport @ composed_local_transport
        row["transport_projection_seconds"] = time.perf_counter() - projection_start
        reference = trial_reference
        current_error = trial_error
        reference_states = trial_states
        if rho > 0.75 and step_norm > 0.8 * radius:
            radius = min(1.5 * radius, args.maximum_radius)
        elif rho < 0.25:
            radius = max(0.5 * radius, args.minimum_radius)

        chart_rebuild_start = time.perf_counter()
        accepted_derivatives = affine_derivatives_at_reference(derivative_template, model, trial_map)
        sector_data, _, _ = build_sector_data(
            model, trial_map, omega1_values, omega3_values, sectors, accepted_derivatives, args
        )
        row["chart_rebuild_seconds"] = time.perf_counter() - chart_rebuild_start
        chart = Chart(trial_map, sector_data)
        reference_states = [data.reference_pathway_states for data in chart.sectors]
        current_error = normalized_squared_error(target_states, reference_states)
        row["iteration_total_seconds"] = time.perf_counter() - iteration_start
        row["cumulative_iterative_seconds"] = time.perf_counter() - iterative_start
        rows.append(row)

    iterative_total_seconds = time.perf_counter() - iterative_start
    final_parameter_error = float(np.linalg.norm(reference - target))
    write_csv(args.output_dir / "iterations.csv", rows)
    exact_target_transport = exact_step_transport(initial_reference_states, target_states, args.rcond)
    exact_recovered_transport = exact_step_transport(
        initial_reference_states, reference_states, args.rcond
    )
    local_composition_error = float(
        np.linalg.norm(composed_local_transport - exact_recovered_transport)
        / max(np.linalg.norm(exact_recovered_transport), 1e-14)
    )
    target_transport_error = float(
        np.linalg.norm(exact_recovered_transport - exact_target_transport)
        / max(np.linalg.norm(exact_target_transport), 1e-14)
    )
    off_diagonal = exact_recovered_transport - np.diag(np.diag(exact_recovered_transport))
    off_diagonal_fraction = float(
        np.linalg.norm(off_diagonal) / max(np.linalg.norm(exact_recovered_transport), 1e-14)
    )
    np.savez_compressed(
        args.output_dir / "accumulated_transport.npz",
        accumulated_transport=exact_recovered_transport,
        composed_local_transport=composed_local_transport,
        exact_recovered_transport=exact_recovered_transport,
        exact_target_transport=exact_target_transport,
    )
    summary = {
        "initial_reference": initial_reference.tolist(),
        "target": target.tolist(),
        "recovered": reference.tolist(),
        "parameter_error_eV": final_parameter_error,
        "final_state_relative_error": float(np.sqrt(current_error)),
        "local_transport_composition_error": local_composition_error,
        "recovered_vs_target_transport_error": target_transport_error,
        "accumulated_off_diagonal_fraction": off_diagonal_fraction,
        "iterations": len(rows),
        "accepted_steps": sum(bool(row["accepted"]) for row in rows),
        "direct_target_evaluations": 1,
        "direct_trial_validations": direct_validations,
        "derivative_cache_seconds": derivative_seconds,
        "initial_chart_seconds": initial_chart_seconds,
        "target_direct_response_seconds": target_direct_seconds,
        "iterative_total_seconds": iterative_total_seconds,
        "optimizer_total_seconds": sum(float(row["optimizer_seconds"]) for row in rows),
        "direct_validation_total_seconds": sum(float(row["direct_validation_seconds"]) for row in rows),
        "transport_projection_total_seconds": sum(float(row["transport_projection_seconds"]) for row in rows),
        "chart_rebuild_total_seconds": sum(float(row["chart_rebuild_seconds"]) for row in rows),
        "final_radius": radius,
        "converged": final_parameter_error <= args.omega0 * args.parameter_tolerance,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
