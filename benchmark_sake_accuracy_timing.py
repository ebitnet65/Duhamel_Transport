#!/usr/bin/env python3
"""Systematic accuracy and amortized-timing benchmark for SAKE-DT.

The direct benchmark evaluates every target pathway with explicit resolvent
inversions.  SAKE constructs one local Duhamel/autodiff expansion at the
reference model and reuses it for all targets.

The output separates three errors that should not be conflated:

1. projection error: loss from representing direct target pathways in the
   reference pathway basis;
2. truncation error: difference between an order-n SAKE transport matrix and
   the directly projected target transport matrix;
3. total signal error: difference between the transported signal and the
   signal obtained by direct target-model inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

import Liouville_autodiff as autodiff
import Liouville_transport_general as general
from exciton_dimer_autodiff import jax_exciton_dimer_liouvillian
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map
from fit_v_system_transport_signal import (
    duhamel_transport_expansion_2d,
    pathway_signal_grid,
    pathway_state_grid_matrix,
    project_transport_derivative,
    relative_error,
    transported_signal,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")


@dataclass(frozen=True)
class SectorData:
    sector: str
    pathways: list[general.CanonicalPathway]
    reference_pathway_signals: np.ndarray
    reference_pathway_states: np.ndarray
    expansion: object


def block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    rows = sum(matrix.shape[0] for matrix in matrices)
    cols = sum(matrix.shape[1] for matrix in matrices)
    out = np.zeros((rows, cols), dtype=complex)
    row0 = 0
    col0 = 0
    for matrix in matrices:
        row1 = row0 + matrix.shape[0]
        col1 = col0 + matrix.shape[1]
        out[row0:row1, col0:col1] = matrix
        row0 = row1
        col0 = col1
    return out


def transported_signal_from_matrix(reference_pathway_signals: np.ndarray, T: np.ndarray) -> np.ndarray:
    pathway_weights = T.T @ np.ones(T.shape[0], dtype=complex)
    return reference_pathway_signals @ pathway_weights


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def load_derivative_cache(path: Path) -> tuple[dict[tuple[int, ...], np.ndarray], float, dict]:
    with np.load(path) as archive:
        derivatives: dict[tuple[int, ...], np.ndarray] = {}
        for name in archive.files:
            key = () if name == "root" else tuple(int(part) for part in name.split("_")[1:])
            derivatives[key] = np.asarray(archive[name], dtype=complex)
    metadata_path = path.with_suffix(".json")
    derivative_seconds = 0.0
    metadata: dict = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        derivative_seconds = float(metadata.get("derivative_seconds", 0.0))
    return derivatives, derivative_seconds, metadata


def validate_derivative_cache(metadata: dict, parameter_map: general.ParameterMap, args: argparse.Namespace) -> None:
    expected = {
        "omega0": args.omega0,
        "delta_omega": args.delta_omega,
        "reference_J": args.reference_J,
        "reference_k": args.reference_k,
        "scale_J": parameter_map.controls[0].scale,
        "scale_k": parameter_map.controls[1].scale,
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(
            "Derivative-cache metadata predates the omega0-scaled parameterization; "
            f"missing {missing}. Regenerate the cache."
        )
    mismatched = {
        key: (metadata[key], value)
        for key, value in expected.items()
        if not np.isclose(float(metadata[key]), float(value))
    }
    if mismatched:
        raise ValueError(f"Derivative cache does not match benchmark parameters: {mismatched}")


def clean_dimer_parameter_map(args: argparse.Namespace) -> general.ParameterMap:
    if args.omega0 <= 0.0:
        raise ValueError("omega0 must be positive")
    if abs(args.delta_omega) >= 2.0 * args.omega0:
        raise ValueError("abs(delta_omega) must be smaller than 2*omega0")
    reference_params = dict(exciton_dimer_parameter_map().reference_params)
    scale_J = args.omega0 if args.delta_J is None else args.delta_J
    scale_k = args.omega0 if args.delta_k is None else args.delta_k
    reference_params.update(
        {
            "omega_1": args.omega0 + 0.5 * args.delta_omega,
            "omega_2": args.omega0 - 0.5 * args.delta_omega,
            "J_12": args.reference_J,
            "k_12": args.reference_k,
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
            general.ControlSpec("J_12", reference_value=args.reference_J, scale=scale_J),
            general.ControlSpec("k_12", reference_value=args.reference_k, scale=scale_k),
        ),
    )


def normalized_direction(args: argparse.Namespace) -> np.ndarray:
    direction = np.asarray([args.direction_J, args.direction_k], dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("The displacement direction must be nonzero")
    return direction / norm


def aggregate_relative_error(actual_blocks: list[np.ndarray], predicted_blocks: list[np.ndarray]) -> float:
    actual = np.concatenate([np.asarray(block).reshape(-1) for block in actual_blocks])
    predicted = np.concatenate([np.asarray(block).reshape(-1) for block in predicted_blocks])
    return relative_error(actual, predicted)


def build_sector_data(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    sectors: tuple[str, ...],
    liouvillian_derivatives: dict[tuple[int, ...], np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[SectorData], float, float]:
    basis_start = time.perf_counter()
    bases: list[tuple[str, list[general.CanonicalPathway], np.ndarray, np.ndarray]] = []
    for sector in sectors:
        candidates = general.canonical_pathways((sector,))
        pathways, _ = general.independent_canonical_pathways(
            model,
            parameter_map,
            omega1_values,
            candidates,
            stage="after_G3",
            tau2=args.tau2,
            eta=args.eta,
            rtol=args.basis_rtol,
        )
        reference_signals = pathway_signal_grid(
            model,
            parameter_map,
            parameter_map.zero(),
            omega1_values,
            omega3_values,
            pathways,
            tau2=args.tau2,
            eta=args.eta,
        )
        reference_states = pathway_state_grid_matrix(
            model,
            parameter_map,
            parameter_map.zero(),
            omega1_values,
            omega3_values,
            pathways,
            tau2=args.tau2,
            eta=args.eta,
        )
        bases.append((sector, pathways, reference_signals, reference_states))
    basis_seconds = time.perf_counter() - basis_start

    expansion_start = time.perf_counter()
    output: list[SectorData] = []
    for sector, pathways, reference_signals, reference_states in bases:
        expansion = duhamel_transport_expansion_2d(
            model,
            parameter_map,
            omega1_values,
            omega3_values,
            pathways,
            liouvillian_derivatives=liouvillian_derivatives,
            tau2=args.tau2,
            eta=args.eta,
            n_quad=args.n_quad,
            max_order=3,
            rcond=args.rcond,
        )
        output.append(
            SectorData(
                sector=sector,
                pathways=pathways,
                reference_pathway_signals=reference_signals,
                reference_pathway_states=reference_states,
                expansion=expansion,
            )
        )
    expansion_seconds = time.perf_counter() - expansion_start
    return output, basis_seconds, expansion_seconds


def accuracy_at_target(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    sector_data: list[SectorData],
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    lam: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    direct_signals: list[np.ndarray] = []
    projected_signals: list[np.ndarray] = []
    direct_states: list[np.ndarray] = []
    projected_states: list[np.ndarray] = []
    exact_transports: list[np.ndarray] = []
    order_transports: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    order_signals: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    direct_start = time.perf_counter()
    for data in sector_data:
        target_signals = pathway_signal_grid(
            model,
            parameter_map,
            lam,
            omega1_values,
            omega3_values,
            data.pathways,
            tau2=args.tau2,
            eta=args.eta,
        )
        target_states = pathway_state_grid_matrix(
            model,
            parameter_map,
            lam,
            omega1_values,
            omega3_values,
            data.pathways,
            tau2=args.tau2,
            eta=args.eta,
        )
        T_exact, _ = project_transport_derivative(
            data.reference_pathway_states,
            target_states,
            rcond=args.rcond,
        )
        direct_signals.append(np.sum(target_signals, axis=1))
        projected_signals.append(
            transported_signal_from_matrix(data.reference_pathway_signals, T_exact)
        )
        direct_states.append(target_states)
        projected_states.append(data.reference_pathway_states @ T_exact.T)
        exact_transports.append(T_exact)

        for order in (1, 2, 3):
            order_transports[order].append(data.expansion.evaluate(lam, order=order))
            order_signals[order].append(
                transported_signal(data.reference_pathway_signals, data.expansion, lam, order=order)
            )
    direct_seconds = time.perf_counter() - direct_start

    direct_signal = sum(direct_signals)
    projected_signal = sum(projected_signals)
    exact_transport = block_diag(exact_transports)
    reference_signal = sum(
        np.sum(data.reference_pathway_signals, axis=1)
        for data in sector_data
    )
    reference_transport = block_diag(
        [data.expansion.evaluate(np.zeros_like(lam), order=3) for data in sector_data]
    )
    off_diagonal_transport = exact_transport - np.diag(np.diag(exact_transport))
    row: dict[str, float] = {
        "lambda_J": float(lam[0]),
        "lambda_k": float(lam[1]),
        "lambda_norm": float(np.linalg.norm(lam)),
        "physical_delta_J": float(parameter_map.controls[0].scale * lam[0]),
        "physical_delta_k": float(parameter_map.controls[1].scale * lam[1]),
        "target_J": float(
            parameter_map.controls[0].reference_value
            + parameter_map.controls[0].scale * lam[0]
        ),
        "target_k": float(
            parameter_map.controls[1].reference_value
            + parameter_map.controls[1].scale * lam[1]
        ),
        "direct_evaluation_seconds": direct_seconds,
        "direct_signal_deformation": relative_error(reference_signal, direct_signal),
        "exact_transport_deformation": relative_error(reference_transport, exact_transport),
        "off_diagonal_transport_fraction": float(
            np.linalg.norm(off_diagonal_transport)
            / max(np.linalg.norm(exact_transport), 1e-14)
        ),
        "state_projection_error": aggregate_relative_error(direct_states, projected_states),
        "signal_projection_error": relative_error(direct_signal, projected_signal),
    }
    for order in (1, 2, 3):
        approx_signal = sum(order_signals[order])
        approx_transport = block_diag(order_transports[order])
        row[f"order{order}_transport_error"] = relative_error(exact_transport, approx_transport)
        row[f"order{order}_signal_total_error"] = relative_error(direct_signal, approx_signal)
        row[f"order{order}_signal_truncation_error"] = relative_error(
            projected_signal,
            approx_signal,
        )
    return row


def direct_signal_at_target(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    sector_data: list[SectorData],
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    lam: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return sum(
        np.sum(
            pathway_signal_grid(
                model,
                parameter_map,
                lam,
                omega1_values,
                omega3_values,
                data.pathways,
                tau2=args.tau2,
                eta=args.eta,
            ),
            axis=1,
        )
        for data in sector_data
    )


def sake_signal_at_target(sector_data: list[SectorData], lam: np.ndarray, order: int) -> np.ndarray:
    return sum(
        transported_signal(data.reference_pathway_signals, data.expansion, lam, order=order)
        for data in sector_data
    )


def median_cumulative_times(
    function,
    targets: list[np.ndarray],
    counts: list[int],
    repeats: int,
) -> dict[int, float]:
    samples: dict[int, list[float]] = {count: [] for count in counts}
    for _ in range(repeats):
        start = time.perf_counter()
        for index, target in enumerate(targets, start=1):
            function(target)
            if index in samples:
                samples[index].append(time.perf_counter() - start)
    return {count: float(np.median(values)) for count, values in samples.items()}


def timing_rows(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    sector_data: list[SectorData],
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    direction: np.ndarray,
    setup_seconds: float,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    counts = sorted(set(args.target_counts))
    max_count = max(counts)
    radii = np.linspace(args.timing_radius_min, args.timing_radius_max, max_count)
    targets = [float(radius) * direction for radius in radii]

    direct = median_cumulative_times(
        lambda lam: direct_signal_at_target(
            model,
            parameter_map,
            sector_data,
            omega1_values,
            omega3_values,
            lam,
            args,
        ),
        targets,
        counts,
        args.timing_repeats,
    )
    sake = median_cumulative_times(
        lambda lam: sake_signal_at_target(sector_data, lam, order=3),
        targets,
        counts,
        max(args.timing_repeats, 3),
    )
    return [
        {
            "n_targets": count,
            "direct_seconds": direct[count],
            "sake_online_seconds": sake[count],
            "sake_amortized_seconds": setup_seconds + sake[count],
            "speedup_online": direct[count] / max(sake[count], 1e-15),
            "speedup_amortized": direct[count] / max(setup_seconds + sake[count], 1e-15),
        }
        for count in counts
    ]


def break_even_targets(rows: list[dict[str, float | int]]) -> int | None:
    for row in rows:
        if float(row["direct_seconds"]) >= float(row["sake_amortized_seconds"]):
            return int(row["n_targets"])
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows: list[dict[str, float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plotted = [row for row in rows if row["lambda_norm"] > 1e-12]
    radius = np.asarray([row["lambda_norm"] for row in plotted])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    markers = {1: "o", 2: "s", 3: "^"}
    for order in (1, 2, 3):
        axes[0].plot(
            radius,
            [row[f"order{order}_transport_error"] for row in plotted],
            marker=markers[order],
            label=f"order {order}",
        )
        axes[1].plot(
            radius,
            [row[f"order{order}_signal_total_error"] for row in plotted],
            marker=markers[order],
            label=f"order {order}",
        )
    axes[1].plot(
        radius,
        [row["signal_projection_error"] for row in plotted],
        color="0.25",
        linestyle="--",
        linewidth=1.4,
        label="projection floor",
    )
    axes[0].set_title("Transport matrix")
    axes[0].set_ylabel("relative error")
    axes[1].set_title("Observable signal")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel(r"dimensionless displacement $\|\lambda\|_2$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"accuracy_vs_displacement.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timing(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    counts = np.asarray([row["n_targets"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(counts, [row["direct_seconds"] for row in rows], marker="o", label="direct inversion")
    ax.plot(
        counts,
        [row["sake_amortized_seconds"] for row in rows],
        marker="s",
        label="SAKE, including setup",
    )
    ax.plot(
        counts,
        [row["sake_online_seconds"] for row in rows],
        marker="^",
        linestyle="--",
        label="SAKE, target evaluation only",
    )
    crossing = break_even_targets(rows)
    if crossing is not None:
        ax.axvline(
            crossing,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
            label=f"break-even: {crossing} targets",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of target parameter points")
    ax.set_ylabel("cumulative wall time / s")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(int(count)) for count in counts])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"wall_time_vs_targets.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(
    output_dir: Path,
    accuracy_rows: list[dict[str, float]],
    timing: list[dict[str, float | int]],
    metadata: dict,
) -> None:
    representative = min(
        accuracy_rows,
        key=lambda row: abs(row["lambda_norm"] - metadata["table_radius"]),
    )
    crossing = break_even_targets(timing)
    table_rows = [
        ("JAX derivatives, cold", metadata["autodiff_cold_seconds"], "s"),
        ("Reference basis and signals", metadata["reference_seconds"], "s"),
        ("Third-order Duhamel expansion", metadata["expansion_seconds"], "s"),
        ("Representative displacement", representative["lambda_norm"], r"$\|\lambda\|_2$"),
        ("Signal projection error", representative["signal_projection_error"], "relative"),
        ("Order-1 transport error", representative["order1_transport_error"], "relative"),
        ("Order-2 transport error", representative["order2_transport_error"], "relative"),
        ("Order-3 transport error", representative["order3_transport_error"], "relative"),
        ("Order-3 total signal error", representative["order3_signal_total_error"], "relative"),
        ("Observed break-even", float(crossing) if crossing is not None else float("nan"), "targets"),
    ]
    with (output_dir / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value", "unit"])
        writer.writerows(table_rows)

    latex_lines = [
        r"\begin{table}",
        r"\caption{Accuracy and timing summary for the SAKE-DT dimer benchmark.}",
        r"\label{tab:sake_benchmark}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcc}",
        r"Quantity & Value & Unit \\",
        r"\hline",
    ]
    for quantity, value, unit in table_rows:
        formatted = f"{value:.4g}" if np.isfinite(value) else "--"
        latex_lines.append(f"{quantity} & {formatted} & {unit} " + r"\\")
    latex_lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"])
    (output_dir / "benchmark_table.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-omega", type=int, default=16)
    parser.add_argument(
        "--omega-min", type=float, default=None,
        help="Lower spectral bound; defaults to omega0 - 0.3 eV.",
    )
    parser.add_argument(
        "--omega-max", type=float, default=None,
        help="Upper spectral bound; defaults to omega0 + 0.3 eV.",
    )
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3e-3)
    parser.add_argument("--omega0", type=float, default=2.0)
    parser.add_argument("--delta-omega", type=float, default=0.18)
    parser.add_argument("--reference-J", type=float, default=0.0, help="Reference J in eV.")
    parser.add_argument(
        "--reference-k", type=float, default=0.0, help="Reference kappa in eV."
    )
    parser.add_argument(
        "--delta-J",
        type=float,
        default=None,
        help="Optional J control scale override; defaults to omega0.",
    )
    parser.add_argument(
        "--delta-k",
        type=float,
        default=None,
        help="Optional kappa control scale override; defaults to omega0.",
    )
    parser.add_argument("--direction-J", type=float, default=-1.0)
    parser.add_argument("--direction-k", type=float, default=1.0)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05),
    )
    parser.add_argument("--target-counts", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--timing-radius-min", type=float, default=0.001)
    parser.add_argument("--timing-radius-max", type=float, default=0.05)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--table-radius",
        type=float,
        default=0.05,
        help="Displacement represented in the manuscript-ready summary table.",
    )
    parser.add_argument("--n-quad", type=int, default=4)
    parser.add_argument(
        "--jax-max-order",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "Highest Liouvillian derivative requested from JAX. The clean dimer "
            "Liouvillian is affine in J_12 and k_12, so order 2 is sufficient for "
            "the full third-order Duhamel transport expansion and supplies the "
            "explicit zero Hessian blocks required by its interface."
        ),
    )
    parser.add_argument(
        "--derivative-cache",
        type=Path,
        default=None,
        help="Load Liouvillian derivative tensors prepared by prepare_sake_dimer_derivatives.py.",
    )
    parser.add_argument("--rcond", type=float, default=1e-11)
    parser.add_argument("--basis-rtol", type=float, default=1e-9)
    parser.add_argument("--clean-gamma", type=float, default=0.01)
    parser.add_argument("--clean-gamma-phi", type=float, default=0.01)
    parser.add_argument("--clean-mu-esa-scale", type=float, default=1.0)
    parser.add_argument("--sectors", nargs="+", choices=("RP", "NRP", "DQC"), default=("RP", "NRP"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dimer_backend_benchmarks/systematic_accuracy_timing"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = exciton_dimer_model()
    parameter_map = clean_dimer_parameter_map(args)
    direction = normalized_direction(args)
    omega_min = args.omega0 - 0.3 if args.omega_min is None else args.omega_min
    omega_max = args.omega0 + 0.3 if args.omega_max is None else args.omega_max
    if omega_min >= omega_max:
        raise ValueError("omega-min must be smaller than omega-max")
    omega1_values = np.linspace(omega_min, omega_max, args.n_omega)
    omega3_values = np.linspace(omega_min, omega_max, args.n_omega)
    sectors = tuple(sector.upper() for sector in args.sectors)

    if args.derivative_cache is None:
        derivative_start = time.perf_counter()
        log(f"computing JAX Liouvillian derivatives through order {args.jax_max_order}")
        liouvillian_derivatives = autodiff.jax_liouvillian_derivatives(
            jax_exciton_dimer_liouvillian,
            parameter_map,
            liouville_dimension=model.liouville_dimension,
            max_order=args.jax_max_order,
        )
        autodiff_cold_seconds = time.perf_counter() - derivative_start
        derivative_source = "in-process JAX"
    else:
        log(f"loading JAX Liouvillian derivatives from {args.derivative_cache}")
        liouvillian_derivatives, autodiff_cold_seconds, cache_metadata = load_derivative_cache(
            args.derivative_cache
        )
        validate_derivative_cache(cache_metadata, parameter_map, args)
        derivative_source = str(args.derivative_cache)
    log("building reference pathway basis and third-order transport expansion")
    sector_data, reference_seconds, expansion_seconds = build_sector_data(
        model,
        parameter_map,
        omega1_values,
        omega3_values,
        sectors,
        liouvillian_derivatives,
        args,
    )
    setup_seconds = autodiff_cold_seconds + reference_seconds + expansion_seconds

    accuracy_rows = []
    for radius in sorted(set(args.radii)):
        log(f"evaluating direct and transported accuracy at radius={radius:g}")
        accuracy_rows.append(accuracy_at_target(
            model,
            parameter_map,
            sector_data,
            omega1_values,
            omega3_values,
            float(radius) * direction,
            args,
        ))
    write_csv(args.output_dir / "accuracy_vs_displacement.csv", accuracy_rows)
    log("running cumulative direct-inversion and SAKE timing")
    timing = timing_rows(
        model,
        parameter_map,
        sector_data,
        omega1_values,
        omega3_values,
        direction,
        setup_seconds,
        args,
    )
    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "n_omega": args.n_omega,
        "grid_points": args.n_omega**2,
        "spectral_window": [omega_min, omega_max],
        "n_quad": args.n_quad,
        "jax_max_order": args.jax_max_order,
        "derivative_source": derivative_source,
        "sectors": sectors,
        "retained_pathways": sum(len(data.pathways) for data in sector_data),
        "reference_basis_condition_numbers": {
            data.sector: float(np.linalg.cond(data.reference_pathway_states))
            for data in sector_data
        },
        "direction": direction.tolist(),
        "omega0": args.omega0,
        "delta_omega": args.delta_omega,
        "site_energies": {
            "omega_1": args.omega0 + 0.5 * args.delta_omega,
            "omega_2": args.omega0 - 0.5 * args.delta_omega,
        },
        "reference_controls": {
            "J_12": args.reference_J,
            "k_12": args.reference_k,
        },
        "control_scales": {
            "J_12": parameter_map.controls[0].scale,
            "k_12": parameter_map.controls[1].scale,
        },
        "table_radius": args.table_radius,
        "autodiff_cold_seconds": autodiff_cold_seconds,
        "reference_seconds": reference_seconds,
        "expansion_seconds": expansion_seconds,
        "setup_seconds": setup_seconds,
        "break_even_targets": break_even_targets(timing),
    }
    write_csv(args.output_dir / "wall_time_vs_targets.csv", timing)
    (args.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    plot_accuracy(accuracy_rows, args.output_dir)
    plot_timing(timing, args.output_dir)
    write_table(args.output_dir, accuracy_rows, timing, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"wrote benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
