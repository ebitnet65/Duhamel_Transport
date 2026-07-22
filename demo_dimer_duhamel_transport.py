#!/usr/bin/env python3
"""
demo_dimer_duhamel_transport.py

Talk-ready proof of concept for Duhamel/transport reconstruction on the
four-level exciton dimer with ESA.

This demo deliberately uses the current finite-difference transport-derivative
backend. The mathematical workflow is the Duhamel transport workflow:

    reference pathway basis at lambda = 0
    transport expansion T(lambda)
    transported pathway-sum signal
    comparison to exact displaced pathway-sum signal

The exact full-commutator spectra are also saved as "model experimental"
spectra for visual context.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import differential_evolution

import Liouville_transport_general as general
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map
from fit_v_system_transport_signal import (
    commutator_signal_grid,
    finite_difference_transport_expansion_2d,
    pathway_state_grid_matrix,
    pathway_signal_grid,
    project_transport_derivative,
    relative_error,
    transported_signal,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")
import matplotlib.pyplot as plt


def log_progress(message: str, start_perf: float) -> None:
    elapsed = time.perf_counter() - start_perf
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp} +{elapsed:9.2f}s] {message}", flush=True)


@dataclass(frozen=True)
class SectorTransportData:
    sector: str
    pathways: list[general.CanonicalPathway]
    candidate_residuals: np.ndarray
    reference_pathway_signals: np.ndarray
    exact_target_pathway_signals: np.ndarray
    expansion: object
    T_exact: np.ndarray
    reconstructed_first: np.ndarray
    reconstructed_second: np.ndarray
    exact_projected_transport_signal: np.ndarray


def plot_transport_matrix(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    output_file: Path,
    *,
    component: str,
) -> None:
    if component == "abs":
        values = np.abs(matrix)
        cmap = "magma"
        label = r"$|T_a^b|$"
        vmin = 0.0
        vmax = float(np.max(values))
    elif component == "real":
        values = np.real(matrix)
        cmap = "RdBu_r"
        label = r"$\mathrm{Re}\,T_a^b$"
        vmax = float(np.max(np.abs(values)))
        vmin = -vmax
    else:
        raise ValueError("component must be 'abs' or 'real'")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("reference basis index b")
    ax.set_ylabel("transported pathway index a")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(np.arange(len(labels)))
    ax.set_yticklabels(np.arange(len(labels)))
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_signal_contours(
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    signal: np.ndarray,
    title: str,
    output_file: Path,
    *,
    component: str,
    levels: int = 18,
) -> None:
    grid = signal.reshape(len(omega3_values), len(omega1_values))
    if component == "real":
        values = np.real(grid)
        vmax = float(np.max(np.abs(values)))
        contour_levels = np.linspace(-vmax, vmax, levels)
        line_levels = np.linspace(-vmax, vmax, 9)
        cmap = "RdBu_r"
        label = r"$\mathrm{Re}\,S$"
    elif component == "abs":
        values = np.abs(grid)
        vmax = float(np.max(values))
        contour_levels = np.linspace(0.0, vmax, levels)
        line_levels = np.linspace(0.0, vmax, 7)[1:]
        cmap = "magma"
        label = r"$|S|$"
    else:
        raise ValueError("component must be 'real' or 'abs'")

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    filled = ax.contourf(
        omega1_values,
        omega3_values,
        values,
        levels=contour_levels,
        cmap=cmap,
        extend="both" if component == "real" else "max",
    )
    if np.max(np.abs(line_levels)) > 0:
        ax.contour(
            omega1_values,
            omega3_values,
            values,
            levels=line_levels,
            colors="k",
            linewidths=0.35,
            alpha=0.35,
        )
    ax.set_xlabel(r"$\omega_1$ / eV")
    ax.set_ylabel(r"$\omega_3$ / eV")
    ax.set_title(title)
    ax.set_xlim(float(omega1_values[0]), float(omega1_values[-1]))
    ax.set_ylim(float(omega3_values[0]), float(omega3_values[-1]))
    fig.colorbar(filled, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(
    lam_values: np.ndarray,
    first_order_errors: np.ndarray,
    second_order_errors: np.ndarray,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(lam_values, first_order_errors, marker="o", label="first order")
    ax.plot(lam_values, second_order_errors, marker="s", label="second order")
    ax.set_xlabel("scale factor along target lambda")
    ax.set_ylabel("relative pathway-signal error")
    ax.set_yscale("log")
    ax.set_title("Transport reconstruction error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_sector_spectrum_triptych(
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    sector: str,
    output_file: Path,
    *,
    component: str,
    order: int,
) -> None:
    panels = (
        ("reference", reference_signal),
        ("target", target_signal),
        (f"transport order {order}", reconstructed_signal),
    )
    values_by_panel = []
    for _label, signal in panels:
        grid = signal.reshape(len(omega3_values), len(omega1_values))
        values_by_panel.append(np.real(grid) if component == "real" else np.abs(grid))

    if component == "real":
        vmax = max(float(np.max(np.abs(values))) for values in values_by_panel)
        levels = np.linspace(-vmax, vmax, 22)
        cmap = "RdBu_r"
        colorbar_label = r"$\mathrm{Re}\,S$"
        extend = "both"
    elif component == "abs":
        vmax = max(float(np.max(values)) for values in values_by_panel)
        levels = np.linspace(0.0, vmax, 22)
        cmap = "magma"
        colorbar_label = r"$|S|$"
        extend = "max"
    else:
        raise ValueError("component must be 'real' or 'abs'")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    filled = None
    for ax, (label, _signal), values in zip(axes, panels, values_by_panel, strict=True):
        filled = ax.contourf(
            omega1_values,
            omega3_values,
            values,
            levels=levels,
            cmap=cmap,
            extend=extend,
        )
        line_levels = np.linspace(-vmax, vmax, 9) if component == "real" else np.linspace(0.0, vmax, 7)[1:]
        if np.max(np.abs(line_levels)) > 0:
            ax.contour(
                omega1_values,
                omega3_values,
                values,
                levels=line_levels,
                colors="k",
                linewidths=0.3,
                alpha=0.3,
            )
        ax.set_title(label)
        ax.set_xlabel(r"$\omega_1$ / eV")
        ax.set_xlim(float(omega1_values[0]), float(omega1_values[-1]))
        ax.set_ylim(float(omega3_values[0]), float(omega3_values[-1]))
    axes[0].set_ylabel(r"$\omega_3$ / eV")
    fig.suptitle(f"{sector} sector spectra")
    if filled is not None:
        fig.colorbar(filled, ax=axes, label=colorbar_label, shrink=0.9)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def exact_transport_matrix_2d(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    pathways: list[general.CanonicalPathway],
    *,
    tau2: float,
    eta: float,
    rcond: float,
) -> np.ndarray:
    P0 = pathway_state_grid_matrix(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
    )
    Plam = pathway_state_grid_matrix(
        model,
        parameter_map,
        lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
    )
    T, _residuals = project_transport_derivative(P0, Plam, rcond=rcond)
    return T


def transported_signal_from_matrix(reference_pathway_signals: np.ndarray, T: np.ndarray) -> np.ndarray:
    pathway_weights = T.T @ np.ones(T.shape[0], dtype=complex)
    return reference_pathway_signals @ pathway_weights


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


def integrated_power(signal: np.ndarray, omega1_values: np.ndarray, omega3_values: np.ndarray) -> float:
    grid = np.abs(signal.reshape(len(omega3_values), len(omega1_values))) ** 2
    inner = np.trapezoid(grid, omega1_values, axis=1)
    return float(np.trapezoid(inner, omega3_values))


def normalize_to_unit_power(
    signal: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    *,
    norm: float | None = None,
) -> tuple[np.ndarray, float]:
    power = integrated_power(signal, omega1_values, omega3_values) if norm is None else norm
    scale = np.sqrt(max(power, 1e-300))
    return signal / scale, power


def sector_signal(data: SectorTransportData, kind: str, *, order: int) -> np.ndarray:
    if kind == "reference":
        return np.sum(data.reference_pathway_signals, axis=1)
    if kind == "target":
        return np.sum(data.exact_target_pathway_signals, axis=1)
    if kind == "reconstructed":
        return data.reconstructed_second if order == 2 else data.reconstructed_first
    raise ValueError("kind must be reference, target, or reconstructed")


def build_sector_transport_data(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    target_lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    sector: str,
    *,
    tau2: float,
    eta: float,
    fd_step: float,
) -> SectorTransportData:
    candidates = general.canonical_pathways((sector,))
    pathways, candidate_residuals = general.independent_canonical_pathways(
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
    expansion = finite_difference_transport_expansion_2d(
        model,
        parameter_map,
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
        step=fd_step,
        rcond=1e-11,
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
        rcond=1e-11,
    )
    return SectorTransportData(
        sector=sector,
        pathways=pathways,
        candidate_residuals=candidate_residuals,
        reference_pathway_signals=reference_pathway_signals,
        exact_target_pathway_signals=exact_target_pathway_signals,
        expansion=expansion,
        T_exact=T_exact,
        reconstructed_first=transported_signal(
            reference_pathway_signals,
            expansion,
            target_lam,
            order=1,
        ),
        reconstructed_second=transported_signal(
            reference_pathway_signals,
            expansion,
            target_lam,
            order=2,
        ),
        exact_projected_transport_signal=transported_signal_from_matrix(
            reference_pathway_signals,
            T_exact,
        ),
    )


def evaluate_sector_reconstruction(
    sector_data: list[SectorTransportData],
    lam: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    return sum(
        transported_signal(data.reference_pathway_signals, data.expansion, lam, order=order)
        for data in sector_data
    )


def exact_sector_pathway_signal(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    sector_data: list[SectorTransportData],
    *,
    tau2: float,
    eta: float,
) -> np.ndarray:
    total = np.zeros(len(omega1_values) * len(omega3_values), dtype=complex)
    for data in sector_data:
        total += np.sum(
            pathway_signal_grid(
                model,
                parameter_map,
                lam,
                omega1_values,
                omega3_values,
                data.pathways,
                tau2=tau2,
                eta=eta,
            ),
            axis=1,
        )
    return total


def correction_table(
    parameter_map: general.ParameterMap,
    input_lam: np.ndarray,
    fit_lam: np.ndarray,
) -> list[dict[str, float | str]]:
    rows = []
    for index, control in enumerate(parameter_map.controls):
        input_delta = control.scale * float(input_lam[index])
        fit_delta = control.scale * float(fit_lam[index])
        rows.append(
            {
                "control": control.name,
                "reference_value": control.reference_value,
                "scale": control.scale,
                "input_lambda": float(input_lam[index]),
                "fit_lambda": float(fit_lam[index]),
                "lambda_error": float(fit_lam[index] - input_lam[index]),
                "input_delta": input_delta,
                "fit_delta": fit_delta,
                "delta_error": fit_delta - input_delta,
                "input_value": control.reference_value + input_delta,
                "fit_value": control.reference_value + fit_delta,
            }
        )
    return rows


def write_correction_rows(handle, rows: list[dict[str, float | str]]) -> None:
    handle.write(
        "control reference scale target_lambda target_delta target_value "
        "fit_lambda fit_delta fit_value lambda_error delta_error\n"
    )
    for row in rows:
        handle.write(
            f"{row['control']} "
            f"{row['reference_value']:.12e} "
            f"{row['scale']:.12e} "
            f"{row['input_lambda']:.12e} "
            f"{row['input_delta']:.12e} "
            f"{row['input_value']:.12e} "
            f"{row['fit_lambda']:.12e} "
            f"{row['fit_delta']:.12e} "
            f"{row['fit_value']:.12e} "
            f"{row['lambda_error']:.12e} "
            f"{row['delta_error']:.12e}\n"
        )


def write_model_description(
    output_file: Path,
    *,
    command: str,
    start_wall: datetime,
    end_wall: datetime,
    elapsed_seconds: float,
    args: argparse.Namespace,
    parameter_map: general.ParameterMap,
    target_lam: np.ndarray,
    params0: dict[str, float],
    params_target: dict[str, float],
    sector_data: list[SectorTransportData],
) -> None:
    output_file.write_text(
        "\n".join(
            [
                "# Exciton Dimer Duhamel Transport Production Run",
                "",
                "## Exact Command",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Timing",
                "",
                f"- Start: `{start_wall.isoformat()}`",
                f"- Stop: `{end_wall.isoformat()}`",
                f"- Elapsed seconds: `{elapsed_seconds:.6f}`",
                "",
                "## Model",
                "",
                "Four-level exciton dimer basis:",
                "",
                "- `|00>`: electronic ground state",
                "- `|10>` and `|01>`: one-exciton local states",
                "- `|11>`: double excitation, included for ESA pathways",
                "",
                "The transition dipole operator couples the ground state to both one-exciton local states and includes ESA coupling from the one-exciton manifold to the double-excited state.",
                "",
                "## Reference And Target Parameters",
                "",
                f"- Reference `J_12 = {params0['J_12']:.12e}`",
                f"- Reference `k_12 = {params0['k_12']:.12e}`",
                f"- Target lambda: `{np.array2string(target_lam, precision=8)}`",
                f"- Target physical displacements: `{parameter_map.physical_displacements(target_lam)}`",
                f"- Target `J_12 = {params_target['J_12']:.12e}`",
                f"- Target `k_12 = {params_target['k_12']:.12e}`",
                "",
                "Control variables are dimensionless displacements about the reference model, with physical parameter values computed as `p(lambda) = p_ref + scale * lambda`.",
                "",
                "## Numerical Setup",
                "",
                f"- Frequency grid: `{args.n_omega} x {args.n_omega}`",
                f"- Frequency window: `{args.omega_min:.8f}` to `{args.omega_max:.8f}` eV",
                f"- Waiting time `tau2`: `{args.tau2:.12e}`",
                f"- Green's-function broadening `eta`: `{args.eta:.12e}`",
                f"- Finite-difference derivative step: `{args.fd_step:.12e}`",
                f"- Transport expansion order used for plotted estimate: `{args.order}`",
                "",
                "## Phase Matching",
                "",
                "Wave-vector conservation is enforced by constructing independent transport blocks for each phase-matching sector. RP and NRP pathway bases are selected, transported, and expanded separately; their signals are only summed after the sector-level transport calculation.",
                "",
                "Sector block sizes:",
                "",
                *[
                    f"- `{data.sector}`: {len(data.pathways)} retained pathways"
                    for data in sector_data
                ],
                "",
                "## Spectrum Normalization",
                "",
                "All saved/plotted spectra are normalized using",
                "",
                "```text",
                "int d omega_1 d omega_3 |S(omega_3, omega_1)|^2 = 1.",
                "```",
                "",
                "For each sector triptych, the reference spectrum is normalized by its own integrated power; the exact target and transported estimate share the exact target sector norm so that their amplitudes are directly comparable.",
                "",
                "## Primary Figure Files",
                "",
                "- `sector_RP_triptych_real.png` and `sector_RP_triptych_abs.png`",
                "- `sector_NRP_triptych_real.png` and `sector_NRP_triptych_abs.png`",
                "- `model_reference_real.png` and `model_reference_abs.png`",
                "- `model_target_real.png` and `model_target_abs.png`",
                "- `pathway_target_reconstructed_order*_real.png` and `pathway_target_reconstructed_order*_abs.png`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dimer Duhamel/transport proof-of-concept demo."
    )
    parser.add_argument("--n-omega", type=int, default=40)
    parser.add_argument("--omega-min", type=float, default=0.8)
    parser.add_argument("--omega-max", type=float, default=1.4)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3e-3)
    parser.add_argument("--fd-step", type=float, default=5e-5)
    parser.add_argument("--lambda-J", type=float, default=-0.25)
    parser.add_argument("--lambda-k", type=float, default=0.25)
    parser.add_argument("--delta-J", type=float, default=0.045)
    parser.add_argument("--delta-k", type=float, default=0.012)
    parser.add_argument("--fit-bound", type=float, default=0.8)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=("RP", "NRP"),
        choices=("RP", "NRP", "DQC"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    start_wall = datetime.now().astimezone()
    start_perf = time.perf_counter()
    args = parse_args()
    if args.output_dir is None:
        stamp = start_wall.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("dimer_duhamel_transport_production") / f"{stamp}_n{args.n_omega}_eta{args.eta:g}"
    command_line = shlex.join(["python", *sys.argv])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"starting run; output directory: {args.output_dir}", start_perf)

    model = exciton_dimer_model()
    parameter_map = exciton_dimer_parameter_map(
        general.ControlSpec("J_12", reference_value=0.0, scale=args.delta_J),
        general.ControlSpec("k_12", reference_value=0.006, scale=args.delta_k),
    )
    target_lam = np.array([args.lambda_J, args.lambda_k], dtype=float)
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    sectors = tuple(sector.upper() for sector in args.sectors)

    log_progress("computing exact reference model spectrum", start_perf)
    reference_model_signal = commutator_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )
    log_progress("computing exact target model spectrum", start_perf)
    target_model_signal = commutator_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )
    log_progress("finished exact model spectra", start_perf)

    sector_data = []
    for sector in sectors:
        log_progress(f"building {sector} sector transport data", start_perf)
        sector_data.append(
            build_sector_transport_data(
                model,
                parameter_map,
                target_lam,
                omega1_values,
                omega3_values,
                sector,
                tau2=args.tau2,
                eta=args.eta,
                fd_step=args.fd_step,
            )
        )
        log_progress(f"finished {sector} sector transport data", start_perf)
    pathways = [pathway for data in sector_data for pathway in data.pathways]
    labels = tuple(pathway.label for pathway in pathways)
    candidate_residuals = np.concatenate([data.candidate_residuals for data in sector_data])
    reference_pathway_signal = sum(
        np.sum(data.reference_pathway_signals, axis=1)
        for data in sector_data
    )
    exact_target_pathway_signal = sum(
        np.sum(data.exact_target_pathway_signals, axis=1)
        for data in sector_data
    )
    reconstructed_first = sum(data.reconstructed_first for data in sector_data)
    reconstructed_second = sum(data.reconstructed_second for data in sector_data)
    selected_reconstruction = reconstructed_second if args.order == 2 else reconstructed_first
    exact_projected_transport_signal = sum(
        data.exact_projected_transport_signal
        for data in sector_data
    )
    residual_signal = selected_reconstruction - exact_target_pathway_signal
    T_exact = block_diag([data.T_exact for data in sector_data])
    T_target = block_diag([
        data.expansion.evaluate(target_lam, order=args.order)
        for data in sector_data
    ])

    def fit_objective(lam: np.ndarray) -> float:
        predicted = evaluate_sector_reconstruction(sector_data, lam, order=args.order)
        return relative_error(exact_target_pathway_signal, predicted) ** 2

    log_progress("starting inverse-fit diagnostic", start_perf)
    fit_result = differential_evolution(
        fit_objective,
        bounds=[(-args.fit_bound, args.fit_bound)] * parameter_map.control_dimension,
        seed=0,
        tol=1e-8,
        atol=1e-12,
        maxiter=200,
        popsize=12,
        polish=True,
        updating="immediate",
    )
    fit_lam = fit_result.x
    fitted_reconstruction = evaluate_sector_reconstruction(sector_data, fit_lam, order=args.order)
    correction_rows = correction_table(parameter_map, target_lam, fit_lam)
    log_progress("finished inverse-fit diagnostic", start_perf)

    log_progress("computing transport error curve", start_perf)
    scan = np.linspace(0.0, 1.0, 6)
    first_errors = []
    second_errors = []
    for scale in scan:
        lam = scale * target_lam
        exact = exact_sector_pathway_signal(
            model,
            parameter_map,
            lam,
            omega1_values,
            omega3_values,
            sector_data,
            tau2=args.tau2,
            eta=args.eta,
        )
        first_errors.append(
            relative_error(
                exact,
                evaluate_sector_reconstruction(sector_data, lam, order=1),
            )
        )
        second_errors.append(
            relative_error(
                exact,
                evaluate_sector_reconstruction(sector_data, lam, order=2),
            )
        )
    first_errors = np.asarray(first_errors)
    second_errors = np.asarray(second_errors)
    log_progress("finished transport error curve", start_perf)

    log_progress("normalizing spectra", start_perf)
    reference_model_signal_norm, reference_model_power = normalize_to_unit_power(
        reference_model_signal,
        omega1_values,
        omega3_values,
    )
    target_model_signal_norm, target_model_power = normalize_to_unit_power(
        target_model_signal,
        omega1_values,
        omega3_values,
    )
    exact_target_pathway_signal_norm, target_pathway_power = normalize_to_unit_power(
        exact_target_pathway_signal,
        omega1_values,
        omega3_values,
    )
    reference_pathway_signal_norm, reference_pathway_power = normalize_to_unit_power(
        reference_pathway_signal,
        omega1_values,
        omega3_values,
    )
    exact_projected_transport_signal_norm, _ = normalize_to_unit_power(
        exact_projected_transport_signal,
        omega1_values,
        omega3_values,
        norm=target_pathway_power,
    )
    selected_reconstruction_norm, _ = normalize_to_unit_power(
        selected_reconstruction,
        omega1_values,
        omega3_values,
        norm=target_pathway_power,
    )
    fitted_reconstruction_norm, _ = normalize_to_unit_power(
        fitted_reconstruction,
        omega1_values,
        omega3_values,
        norm=target_pathway_power,
    )
    residual_signal_norm = selected_reconstruction_norm - exact_target_pathway_signal_norm

    log_progress("writing sector spectra", start_perf)
    sector_npz_payload = {}
    for data in sector_data:
        sector_reference = sector_signal(data, "reference", order=args.order)
        sector_target = sector_signal(data, "target", order=args.order)
        sector_reconstructed = sector_signal(data, "reconstructed", order=args.order)
        sector_reference_norm, sector_reference_power = normalize_to_unit_power(
            sector_reference,
            omega1_values,
            omega3_values,
        )
        sector_target_norm, sector_target_power = normalize_to_unit_power(
            sector_target,
            omega1_values,
            omega3_values,
        )
        sector_reconstructed_norm, _ = normalize_to_unit_power(
            sector_reconstructed,
            omega1_values,
            omega3_values,
            norm=sector_target_power,
        )
        sector_key = data.sector.lower()
        sector_npz_payload[f"sector_{sector_key}_reference_signal"] = sector_reference
        sector_npz_payload[f"sector_{sector_key}_target_signal"] = sector_target
        sector_npz_payload[f"sector_{sector_key}_reconstructed_signal"] = sector_reconstructed
        sector_npz_payload[f"sector_{sector_key}_reference_signal_normalized"] = sector_reference_norm
        sector_npz_payload[f"sector_{sector_key}_target_signal_normalized"] = sector_target_norm
        sector_npz_payload[f"sector_{sector_key}_reconstructed_signal_normalized"] = sector_reconstructed_norm
        sector_npz_payload[f"sector_{sector_key}_normalization_powers"] = np.asarray([
            sector_reference_power,
            sector_target_power,
        ])

        for component in ("real", "abs"):
            plot_sector_spectrum_triptych(
                omega1_values,
                omega3_values,
                sector_reference_norm,
                sector_target_norm,
                sector_reconstructed_norm,
                data.sector,
                args.output_dir / f"sector_{data.sector}_triptych_{component}.png",
                component=component,
                order=args.order,
            )
            plot_signal_contours(
                omega1_values,
                omega3_values,
                sector_reference_norm,
                f"{data.sector} reference",
                args.output_dir / f"sector_{data.sector}_reference_{component}.png",
                component=component,
            )
            plot_signal_contours(
                omega1_values,
                omega3_values,
                sector_target_norm,
                f"{data.sector} target",
                args.output_dir / f"sector_{data.sector}_target_{component}.png",
                component=component,
            )
            plot_signal_contours(
                omega1_values,
                omega3_values,
                sector_reconstructed_norm,
                f"{data.sector} transported estimate, order {args.order}",
                args.output_dir / f"sector_{data.sector}_distorted_estimate_order{args.order}_{component}.png",
                component=component,
            )

    log_progress("writing combined spectra", start_perf)
    for component in ("real", "abs"):
        plot_signal_contours(
            omega1_values,
            omega3_values,
            reference_model_signal_norm,
            "Exact model spectrum at reference",
            args.output_dir / f"model_reference_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            target_model_signal_norm,
            "Exact model spectrum at target",
            args.output_dir / f"model_target_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            exact_target_pathway_signal_norm,
            "Exact target pathway signal",
            args.output_dir / f"pathway_target_exact_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            exact_projected_transport_signal_norm,
            "Exact projected transport",
            args.output_dir / f"pathway_target_exact_projected_transport_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            selected_reconstruction_norm,
            f"Transport reconstruction, order {args.order}",
            args.output_dir / f"pathway_target_reconstructed_order{args.order}_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            residual_signal_norm,
            "Transport residual",
            args.output_dir / f"pathway_transport_residual_order{args.order}_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            fitted_reconstruction_norm,
            f"Fitted transport reconstruction, order {args.order}",
            args.output_dir / f"pathway_target_fitted_reconstruction_order{args.order}_{component}.png",
            component=component,
        )

    plot_transport_matrix(
        T_target,
        labels,
        f"Transport matrix T(lambda), order {args.order}",
        args.output_dir / f"transport_matrix_order{args.order}_abs.png",
        component="abs",
    )
    plot_transport_matrix(
        T_target,
        labels,
        f"Transport matrix T(lambda), order {args.order}",
        args.output_dir / f"transport_matrix_order{args.order}_real.png",
        component="real",
    )
    plot_error_curve(
        scan,
        first_errors,
        second_errors,
        args.output_dir / "transport_error_curve.png",
    )
    log_progress("finished plots", start_perf)

    params0 = parameter_map.params(parameter_map.zero())
    params_target = parameter_map.params(target_lam)
    T0_block = block_diag([data.expansion.T0 for data in sector_data])
    end_wall = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - start_perf
    log_progress("writing summary and data archive", start_perf)
    summary_file = args.output_dir / "demo_summary.txt"
    with summary_file.open("w", encoding="utf-8") as handle:
        handle.write("Dimer Duhamel/transport proof of concept\n")
        handle.write("=" * 72 + "\n\n")
        handle.write("Scope\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            "The TARGET section gives the exact input parameters used to generate "
            "the displaced model spectra.\n"
        )
        handle.write(
            "The INVERSE-FIT section is a separate diagnostic that asks what "
            "correction values are recovered by minimizing the transported "
            "pathway-signal residual.\n\n"
        )

        handle.write("Numerical setup\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"command: {command_line}\n")
        handle.write(f"output directory: {args.output_dir}\n")
        handle.write(f"start time: {start_wall.isoformat()}\n")
        handle.write(f"stop time: {end_wall.isoformat()}\n")
        handle.write(f"elapsed seconds: {elapsed_seconds:.6f}\n")
        handle.write("Derivative backend: finite-difference pathway transport\n")
        handle.write(f"sectors: {list(sectors)}\n")
        handle.write("wave-vector conservation: enforced by independent sector blocks\n")
        handle.write(f"n_omega: {args.n_omega}\n")
        handle.write(f"omega window: {args.omega_min:.8f} to {args.omega_max:.8f} eV\n")
        handle.write(f"eta: {args.eta:.12e}\n\n")

        handle.write("Spectral normalization\n")
        handle.write("-" * 72 + "\n")
        handle.write("Plotted and saved normalized spectra use int d omega_1 d omega_3 |S|^2 = 1.\n")
        handle.write(
            "Pathway reconstructions and pathway residuals are normalized by the "
            "exact target pathway-signal norm so amplitudes remain comparable.\n"
        )
        handle.write(f"reference model raw integrated power: {reference_model_power:.12e}\n")
        handle.write(f"target model raw integrated power: {target_model_power:.12e}\n")
        handle.write(f"reference pathway raw integrated power: {reference_pathway_power:.12e}\n")
        handle.write(f"target pathway raw integrated power: {target_pathway_power:.12e}\n\n")

        handle.write("Reference model\n")
        handle.write("-" * 72 + "\n")
        handle.write("lambda_reference = [0, 0]\n")
        handle.write(f"reference params J_12={params0['J_12']:.12e}, k_12={params0['k_12']:.12e}\n")
        handle.write("J_12 is zero in the reference model by construction.\n\n")

        handle.write("Injected target corrections used to generate target spectra\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"target lambda: {np.array2string(target_lam, precision=8)}\n")
        handle.write(f"target physical displacements: {parameter_map.physical_displacements(target_lam)}\n")
        handle.write(
            f"target params J_12={params_target['J_12']:.12e}, "
            f"k_12={params_target['k_12']:.12e}\n\n"
        )

        handle.write("Input target versus inverse-fit diagnostic\n")
        handle.write("-" * 72 + "\n")
        handle.write(
            "The target columns below are the exact injected corrections. "
            "The fit columns are inferred corrections from an inverse residual "
            "minimization and need not equal the injected values if the inverse "
            "metric is not uniquely conditioned.\n"
        )
        write_correction_rows(handle, correction_rows)
        handle.write("\n")
        handle.write(f"fit lambda: {np.array2string(fit_lam, precision=8)}\n")
        handle.write(f"fit physical displacements: {parameter_map.physical_displacements(fit_lam)}\n")
        handle.write(f"lambda fit norm error: {np.linalg.norm(fit_lam - target_lam):.12e}\n")
        handle.write(f"fit objective: {fit_result.fun:.12e}\n")
        handle.write(f"fit success: {fit_result.success}\n")
        handle.write(f"fit message: {fit_result.message}\n\n")

        handle.write("Transport reconstruction diagnostics at injected target lambda\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"number of retained pathways: {len(pathways)}\n")
        handle.write(f"candidate residual max: {np.max(candidate_residuals):.12e}\n")
        handle.write(f"||T0-I||, block diagonal sectors: {np.linalg.norm(T0_block - np.eye(len(pathways))):.12e}\n")
        handle.write("sector block sizes\n")
        for data in sector_data:
            handle.write(
                f"  {data.sector}: retained_pathways={len(data.pathways)} "
                f"candidate_residual_max={np.max(data.candidate_residuals):.12e}\n"
            )
        handle.write(
            "exact projected-transport target relative error: "
            f"{relative_error(exact_target_pathway_signal, exact_projected_transport_signal):.12e}\n"
        )
        handle.write(
            "first-order target relative error: "
            f"{relative_error(exact_target_pathway_signal, reconstructed_first):.12e}\n"
        )
        handle.write(
            "second-order target relative error: "
            f"{relative_error(exact_target_pathway_signal, reconstructed_second):.12e}\n"
        )
        handle.write(
            "fitted transport target relative error: "
            f"{relative_error(exact_target_pathway_signal, fitted_reconstruction):.12e}\n"
        )
        handle.write(
            "reference-to-target pathway change: "
            f"{relative_error(exact_target_pathway_signal, reference_pathway_signal):.12e}\n\n"
        )
        handle.write("pathway labels\n")
        handle.write("-" * 72 + "\n")
        for index, label in enumerate(labels):
            handle.write(f"{index:2d}: {label}\n")

    write_model_description(
        args.output_dir / "model_description.md",
        command=command_line,
        start_wall=start_wall,
        end_wall=end_wall,
        elapsed_seconds=elapsed_seconds,
        args=args,
        parameter_map=parameter_map,
        target_lam=target_lam,
        params0=params0,
        params_target=params_target,
        sector_data=sector_data,
    )

    np.savez_compressed(
        args.output_dir / "demo_dimer_duhamel_transport.npz",
        omega1_values=omega1_values,
        omega3_values=omega3_values,
        target_lam=target_lam,
        reference_model_signal=reference_model_signal,
        target_model_signal=target_model_signal,
        reference_model_signal_normalized=reference_model_signal_norm,
        target_model_signal_normalized=target_model_signal_norm,
        reference_pathway_signal=reference_pathway_signal,
        exact_target_pathway_signal=exact_target_pathway_signal,
        reference_pathway_signal_normalized=reference_pathway_signal_norm,
        exact_target_pathway_signal_normalized=exact_target_pathway_signal_norm,
        exact_projected_transport_signal=exact_projected_transport_signal,
        exact_projected_transport_signal_normalized=exact_projected_transport_signal_norm,
        reconstructed_first=reconstructed_first,
        reconstructed_second=reconstructed_second,
        selected_reconstruction_normalized=selected_reconstruction_norm,
        fitted_reconstruction=fitted_reconstruction,
        fitted_reconstruction_normalized=fitted_reconstruction_norm,
        residual_signal_normalized=residual_signal_norm,
        normalization_powers=np.asarray([
            reference_model_power,
            target_model_power,
            reference_pathway_power,
            target_pathway_power,
        ]),
        normalization_power_labels=np.asarray([
            "reference_model",
            "target_model",
            "reference_pathway",
            "target_pathway",
        ], dtype=str),
        T_exact=T_exact,
        T_target=T_target,
        T0_block=T0_block,
        fit_lam=fit_lam,
        correction_control_names=np.asarray([row["control"] for row in correction_rows], dtype=str),
        correction_table=np.asarray([
            [
                row["reference_value"],
                row["scale"],
                row["input_lambda"],
                row["fit_lambda"],
                row["lambda_error"],
                row["input_delta"],
                row["fit_delta"],
                row["delta_error"],
                row["input_value"],
                row["fit_value"],
            ]
            for row in correction_rows
        ], dtype=float),
        correction_table_columns=np.asarray([
            "reference_value",
            "scale",
            "input_lambda",
            "fit_lambda",
            "lambda_error",
            "input_delta",
            "fit_delta",
            "delta_error",
            "input_value",
            "fit_value",
        ], dtype=str),
        labels=np.asarray(labels, dtype=str),
        sector_labels=np.asarray([data.sector for data in sector_data], dtype=str),
        sector_block_sizes=np.asarray([len(data.pathways) for data in sector_data], dtype=int),
        scan=scan,
        first_errors=first_errors,
        second_errors=second_errors,
        start_time=np.asarray(start_wall.isoformat(), dtype=str),
        stop_time=np.asarray(end_wall.isoformat(), dtype=str),
        elapsed_seconds=np.asarray(elapsed_seconds),
        command=np.asarray(command_line, dtype=str),
        **sector_npz_payload,
    )

    print("Dimer Duhamel/transport proof of concept")
    print("=" * 72)
    print(f"target lambda: {np.array2string(target_lam, precision=8)}")
    print(f"physical displacements: {parameter_map.physical_displacements(target_lam)}")
    print(f"fit lambda   : {np.array2string(fit_lam, precision=8)}")
    print(f"fit displacements: {parameter_map.physical_displacements(fit_lam)}")
    print(f"lambda fit norm error: {np.linalg.norm(fit_lam - target_lam):.6e}")
    print(f"retained pathways: {len(pathways)}")
    print(
        "relative error, exact projected transport: "
        f"{relative_error(exact_target_pathway_signal, exact_projected_transport_signal):.6e}"
    )
    print(
        "relative error, first order: "
        f"{relative_error(exact_target_pathway_signal, reconstructed_first):.6e}"
    )
    print(
        "relative error, second order: "
        f"{relative_error(exact_target_pathway_signal, reconstructed_second):.6e}"
    )
    print(
        "relative error, fitted transport: "
        f"{relative_error(exact_target_pathway_signal, fitted_reconstruction):.6e}"
    )
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
