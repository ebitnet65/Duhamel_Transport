#!/usr/bin/env python3
"""
benchmark_dimer_transport_backends.py

Compare finite-difference pathway transport against Duhamel transport using
autodiff Liouvillian derivatives for the clean four-level dimer model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import shlex
import sys
import time

import numpy as np

import Liouville_autodiff as autodiff
import Liouville_transport_general as general
from demo_dimer_duhamel_transport import (
    block_diag,
    exact_transport_matrix_2d,
    relative_error,
    transported_signal_from_matrix,
)
from exciton_dimer_autodiff import jax_exciton_dimer_liouvillian
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map
from fit_v_system_transport_signal import (
    duhamel_transport_expansion_2d,
    finite_difference_transport_expansion_2d,
    pathway_signal_grid,
    transported_signal,
)


@dataclass(frozen=True)
class SectorBasis:
    sector: str
    pathways: list[general.CanonicalPathway]
    reference_pathway_signals: np.ndarray
    exact_target_pathway_signals: np.ndarray
    T_exact: np.ndarray


def clean_dimer_parameter_map(args: argparse.Namespace) -> general.ParameterMap:
    reference_params = dict(exciton_dimer_parameter_map().reference_params)
    reference_params.update(
        {
            "J_12": 0.0,
            "k_12": 0.0,
            "anharmonicity": 0.0,
            "gamma_1": args.clean_gamma,
            "gamma_2": args.clean_gamma,
            "gamma_phi_1": args.clean_gamma_phi,
            "gamma_phi_2": args.clean_gamma_phi,
            "mu_1": 1.0,
            "mu_2": 1.0,
            "mu_esa_scale": args.clean_mu_esa_scale,
        }
    )
    return general.ParameterMap(
        reference_params=reference_params,
        controls=(
            general.ControlSpec("J_12", reference_value=0.0, scale=args.delta_J),
            general.ControlSpec("k_12", reference_value=0.0, scale=args.delta_k),
        ),
    )


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def build_sector_basis(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    target_lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    sector: str,
    *,
    tau2: float,
    eta: float,
    rcond: float,
) -> SectorBasis:
    candidates = general.canonical_pathways((sector,))
    pathways, _candidate_residuals = general.independent_canonical_pathways(
        model,
        parameter_map,
        omega1_values,
        candidates,
        stage="after_G3",
        tau2=tau2,
        eta=eta,
        rtol=1e-9,
    )
    reference_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
    )
    exact_target_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
    )
    T_exact = exact_transport_matrix_2d(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
        rcond=rcond,
    )
    return SectorBasis(
        sector=sector,
        pathways=pathways,
        reference_pathway_signals=reference_pathway_signals,
        exact_target_pathway_signals=exact_target_pathway_signals,
        T_exact=T_exact,
    )


def reconstruct_signal(
    sector_bases: list[SectorBasis],
    expansions,
    lam: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    return sum(
        transported_signal(basis.reference_pathway_signals, expansion, lam, order=order)
        for basis, expansion in zip(sector_bases, expansions, strict=True)
    )


def benchmark_backend(
    backend: str,
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    sector_bases: list[SectorBasis],
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    target_lam: np.ndarray,
    args: argparse.Namespace,
    liouvillian_derivatives: dict[tuple[int, ...], np.ndarray] | None,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    expansions = []
    for basis in sector_bases:
        if backend == "finite_difference":
            expansion = finite_difference_transport_expansion_2d(
                model,
                parameter_map,
                omega1_values,
                omega3_values,
                basis.pathways,
                tau2=args.tau2,
                eta=args.eta,
                step=args.fd_step,
                second_step=args.fd_second_step,
                rcond=args.rcond,
            )
        elif backend == "duhamel_autodiff":
            if liouvillian_derivatives is None:
                raise ValueError("liouvillian_derivatives are required for duhamel_autodiff")
            expansion = duhamel_transport_expansion_2d(
                model,
                parameter_map,
                omega1_values,
                omega3_values,
                basis.pathways,
                liouvillian_derivatives=liouvillian_derivatives,
                tau2=args.tau2,
                eta=args.eta,
                n_quad=args.n_quad,
                max_order=args.max_order,
                rcond=args.rcond,
            )
        else:
            raise ValueError(f"Unknown backend {backend}")
        expansions.append(expansion)
    backend_seconds = time.perf_counter() - start

    exact_target = sum(np.sum(basis.exact_target_pathway_signals, axis=1) for basis in sector_bases)
    exact_projected = sum(
        transported_signal_from_matrix(basis.reference_pathway_signals, basis.T_exact)
        for basis in sector_bases
    )
    reconstructed_order1 = reconstruct_signal(
        sector_bases,
        expansions,
        target_lam,
        order=1,
    )
    reconstructed_order2 = reconstruct_signal(
        sector_bases,
        expansions,
        target_lam,
        order=2,
    )
    reconstructed_order3 = reconstruct_signal(
        sector_bases,
        expansions,
        target_lam,
        order=3,
    )
    T_exact = block_diag([basis.T_exact for basis in sector_bases])
    T_order1 = block_diag([expansion.evaluate(target_lam, order=1) for expansion in expansions])
    T_order2 = block_diag([expansion.evaluate(target_lam, order=2) for expansion in expansions])
    T_order3 = block_diag([expansion.evaluate(target_lam, order=3) for expansion in expansions])

    return {
        "backend": backend,
        "backend_seconds": backend_seconds,
        "order1_error": relative_error(exact_target, reconstructed_order1),
        "order2_error": relative_error(exact_target, reconstructed_order2),
        "order3_error": relative_error(exact_target, reconstructed_order3),
        "exact_projected_error": relative_error(exact_target, exact_projected),
        "order1_T_rel_error": relative_error(T_exact.reshape(-1), T_order1.reshape(-1)),
        "order2_T_rel_error": relative_error(T_exact.reshape(-1), T_order2.reshape(-1)),
        "order3_T_rel_error": relative_error(T_exact.reshape(-1), T_order3.reshape(-1)),
        "order1_T_norm_ratio": np.linalg.norm(T_order1) / max(np.linalg.norm(T_exact), 1e-14),
        "order2_T_norm_ratio": np.linalg.norm(T_order2) / max(np.linalg.norm(T_exact), 1e-14),
        "order3_T_norm_ratio": np.linalg.norm(T_order3) / max(np.linalg.norm(T_exact), 1e-14),
        "retained_pathways": sum(len(basis.pathways) for basis in sector_bases),
    }


def estimate_power_law(rows: list[dict[str, float | int | str]], backend: str) -> float:
    selected = [
        row for row in rows
        if row["backend"] == backend and float(row["backend_seconds"]) > 0
    ]
    if len(selected) < 2:
        return float("nan")
    x = np.log(np.asarray([int(row["n_omega"]) for row in selected], dtype=float))
    y = np.log(np.asarray([float(row["backend_seconds"]) for row in selected], dtype=float))
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark finite-difference versus autodiff/Duhamel dimer transport."
    )
    parser.add_argument("--n-omega", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--omega-min", type=float, default=0.8)
    parser.add_argument("--omega-max", type=float, default=1.4)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3e-3)
    parser.add_argument("--lambda-J", type=float, default=-0.25)
    parser.add_argument("--lambda-k", type=float, default=0.25)
    parser.add_argument("--delta-J", type=float, default=0.045)
    parser.add_argument("--delta-k", type=float, default=0.012)
    parser.add_argument("--clean-gamma", type=float, default=0.01)
    parser.add_argument("--clean-gamma-phi", type=float, default=0.01)
    parser.add_argument("--clean-mu-esa-scale", type=float, default=1.0)
    parser.add_argument("--fd-step", type=float, default=5e-5)
    parser.add_argument("--fd-second-step", type=float, default=1e-2)
    parser.add_argument("--n-quad", type=int, default=16)
    parser.add_argument("--max-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--rcond", type=float, default=1e-11)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("finite_difference", "duhamel_autodiff"),
        default=("finite_difference", "duhamel_autodiff"),
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        choices=("RP", "NRP", "DQC"),
        default=("RP", "NRP"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_wall = datetime.now().astimezone()
    if args.output_dir is None:
        stamp = start_wall.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("dimer_backend_benchmarks") / f"{stamp}_autodiff_compare"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["python", *sys.argv])

    model = exciton_dimer_model()
    parameter_map = clean_dimer_parameter_map(args)
    target_lam = np.array([args.lambda_J, args.lambda_k], dtype=float)
    sectors = tuple(sector.upper() for sector in args.sectors)

    log("computing autodiff Liouvillian derivatives")
    derivative_start = time.perf_counter()
    liouvillian_derivatives = autodiff.jax_liouvillian_derivatives(
        jax_exciton_dimer_liouvillian,
        parameter_map,
        liouville_dimension=model.liouville_dimension,
        max_order=3,
    )
    derivative_seconds = time.perf_counter() - derivative_start
    log(f"autodiff Liouvillian derivatives ready in {derivative_seconds:.3f}s")

    rows: list[dict[str, float | int | str]] = []
    for n_omega in args.n_omega:
        omega1_values = np.linspace(args.omega_min, args.omega_max, n_omega)
        omega3_values = np.linspace(args.omega_min, args.omega_max, n_omega)
        log(f"building common sector data for n_omega={n_omega}")
        common_start = time.perf_counter()
        sector_bases = [
            build_sector_basis(
                model,
                parameter_map,
                target_lam,
                omega1_values,
                omega3_values,
                sector,
                tau2=args.tau2,
                eta=args.eta,
                rcond=args.rcond,
            )
            for sector in sectors
        ]
        common_seconds = time.perf_counter() - common_start
        log(f"common sector data ready for n_omega={n_omega} in {common_seconds:.3f}s")

        for backend in args.backends:
            log(f"running backend={backend} n_omega={n_omega}")
            row = benchmark_backend(
                backend,
                model,
                parameter_map,
                sector_bases,
                omega1_values,
                omega3_values,
                target_lam,
                args,
                liouvillian_derivatives,
            )
            row.update(
                {
                    "n_omega": n_omega,
                    "grid_points": n_omega * n_omega,
                    "common_seconds": common_seconds,
                    "autodiff_derivative_seconds": derivative_seconds,
                    "n_quad": args.n_quad,
                    "max_order": args.max_order,
                }
            )
            rows.append(row)
            log(
                f"{backend} n={n_omega}: time={row['backend_seconds']:.3f}s "
                f"err1={row['order1_error']:.4e} err2={row['order2_error']:.4e} "
                f"err3={row['order3_error']:.4e}"
            )

    csv_file = args.output_dir / "benchmark_summary.csv"
    fieldnames = [
        "backend",
        "n_omega",
        "grid_points",
        "retained_pathways",
        "backend_seconds",
        "common_seconds",
        "autodiff_derivative_seconds",
        "n_quad",
        "max_order",
        "exact_projected_error",
        "order1_error",
        "order2_error",
        "order3_error",
        "order1_T_rel_error",
        "order2_T_rel_error",
        "order3_T_rel_error",
        "order1_T_norm_ratio",
        "order2_T_norm_ratio",
        "order3_T_norm_ratio",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    power_laws = {
        backend: estimate_power_law(rows, backend)
        for backend in args.backends
    }
    summary = {
        "command": command,
        "start_time": start_wall.isoformat(),
        "stop_time": datetime.now().astimezone().isoformat(),
        "target_lambda": target_lam.tolist(),
        "sectors": list(sectors),
        "n_omega": list(args.n_omega),
        "backends": list(args.backends),
        "autodiff_derivative_seconds": derivative_seconds,
        "timing_power_law_vs_n_omega": power_laws,
        "rows": rows,
    }
    json_file = args.output_dir / "benchmark_summary.json"
    json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_file = args.output_dir / "benchmark_summary.md"
    lines = [
        "# Dimer Transport Backend Benchmark",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Timing Scaling",
        "",
    ]
    for backend, slope in power_laws.items():
        lines.append(f"- `{backend}`: backend time scales approximately as `n_omega^{slope:.2f}`")
    lines.extend(["", "## Results", ""])
    lines.append(
        "| backend | n_omega | backend s | exact projected err | order 1 err | order 2 err | order 3 err |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['n_omega']} | {row['backend_seconds']:.3f} | "
            f"{row['exact_projected_error']:.4e} | {row['order1_error']:.4e} | "
            f"{row['order2_error']:.4e} | {row['order3_error']:.4e} |"
        )
    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"wrote {csv_file}")
    log(f"wrote {json_file}")
    log(f"wrote {md_file}")


if __name__ == "__main__":
    main()
