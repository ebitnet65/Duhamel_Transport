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

import numpy as np

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
        default=Path("dimer_duhamel_transport_demo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = exciton_dimer_model()
    parameter_map = exciton_dimer_parameter_map(
        general.ControlSpec("J_12", reference_value=0.0, scale=args.delta_J),
        general.ControlSpec("k_12", reference_value=0.006, scale=args.delta_k),
    )
    target_lam = np.array([args.lambda_J, args.lambda_k], dtype=float)
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    sectors = tuple(sector.upper() for sector in args.sectors)

    candidates = general.canonical_pathways(sectors)
    pathways, candidate_residuals = general.independent_canonical_pathways(
        model,
        parameter_map,
        omega1_values,
        candidates,
        stage="after_G3",
        tau2=args.tau2,
        eta=args.eta,
        rtol=1e-9,
    )

    reference_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
    )
    exact_target_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
    )
    reference_pathway_signal = np.sum(reference_pathway_signals, axis=1)
    exact_target_pathway_signal = np.sum(exact_target_pathway_signals, axis=1)

    reference_model_signal = commutator_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )
    target_model_signal = commutator_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )

    expansion = finite_difference_transport_expansion_2d(
        model,
        parameter_map,
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
        step=args.fd_step,
        rcond=1e-11,
    )
    reconstructed_first = transported_signal(
        reference_pathway_signals,
        expansion,
        target_lam,
        order=1,
    )
    reconstructed_second = transported_signal(
        reference_pathway_signals,
        expansion,
        target_lam,
        order=2,
    )
    selected_reconstruction = reconstructed_second if args.order == 2 else reconstructed_first
    T_exact = exact_transport_matrix_2d(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
        rcond=1e-11,
    )
    exact_projected_transport_signal = transported_signal_from_matrix(
        reference_pathway_signals,
        T_exact,
    )
    residual_signal = selected_reconstruction - exact_target_pathway_signal
    T_target = expansion.evaluate(target_lam, order=args.order)

    scan = np.linspace(0.0, 1.0, 6)
    first_errors = []
    second_errors = []
    for scale in scan:
        lam = scale * target_lam
        exact = np.sum(
            pathway_signal_grid(
                model,
                parameter_map,
                lam,
                omega1_values,
                omega3_values,
                pathways,
                tau2=args.tau2,
                eta=args.eta,
            ),
            axis=1,
        )
        first_errors.append(
            relative_error(
                exact,
                transported_signal(reference_pathway_signals, expansion, lam, order=1),
            )
        )
        second_errors.append(
            relative_error(
                exact,
                transported_signal(reference_pathway_signals, expansion, lam, order=2),
            )
        )
    first_errors = np.asarray(first_errors)
    second_errors = np.asarray(second_errors)

    for component in ("real", "abs"):
        plot_signal_contours(
            omega1_values,
            omega3_values,
            reference_model_signal,
            "Exact model spectrum at reference",
            args.output_dir / f"model_reference_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            target_model_signal,
            "Exact model spectrum at target",
            args.output_dir / f"model_target_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            exact_target_pathway_signal,
            "Exact target pathway signal",
            args.output_dir / f"pathway_target_exact_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            exact_projected_transport_signal,
            "Exact projected transport",
            args.output_dir / f"pathway_target_exact_projected_transport_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            selected_reconstruction,
            f"Transport reconstruction, order {args.order}",
            args.output_dir / f"pathway_target_reconstructed_order{args.order}_{component}.png",
            component=component,
        )
        plot_signal_contours(
            omega1_values,
            omega3_values,
            residual_signal,
            "Transport residual",
            args.output_dir / f"pathway_transport_residual_order{args.order}_{component}.png",
            component=component,
        )

    plot_transport_matrix(
        T_target,
        expansion.labels,
        f"Transport matrix T(lambda), order {args.order}",
        args.output_dir / f"transport_matrix_order{args.order}_abs.png",
        component="abs",
    )
    plot_transport_matrix(
        T_target,
        expansion.labels,
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

    params0 = parameter_map.params(parameter_map.zero())
    params_target = parameter_map.params(target_lam)
    summary_file = args.output_dir / "demo_summary.txt"
    with summary_file.open("w", encoding="utf-8") as handle:
        handle.write("Dimer Duhamel/transport proof of concept\n")
        handle.write("=" * 72 + "\n\n")
        handle.write("Derivative backend: finite-difference pathway transport\n")
        handle.write(f"sectors: {list(sectors)}\n")
        handle.write(f"n_omega: {args.n_omega}\n")
        handle.write(f"omega window: {args.omega_min:.8f} to {args.omega_max:.8f} eV\n")
        handle.write(f"eta: {args.eta:.12e}\n")
        handle.write("reference model is fixed at lambda=0, with J_12=0 by construction\n")
        handle.write(f"target lambda: {np.array2string(target_lam, precision=8)}\n")
        handle.write(f"physical displacements: {parameter_map.physical_displacements(target_lam)}\n")
        handle.write(f"reference params J_12={params0['J_12']:.12e}, k_12={params0['k_12']:.12e}\n")
        handle.write(
            f"target params J_12={params_target['J_12']:.12e}, "
            f"k_12={params_target['k_12']:.12e}\n\n"
        )
        handle.write(f"number of retained pathways: {len(pathways)}\n")
        handle.write(f"candidate residual max: {np.max(candidate_residuals):.12e}\n")
        handle.write(f"||T0-I||: {np.linalg.norm(expansion.T0 - np.eye(len(pathways))):.12e}\n")
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
            "reference-to-target pathway change: "
            f"{relative_error(exact_target_pathway_signal, reference_pathway_signal):.12e}\n\n"
        )
        handle.write("pathway labels\n")
        handle.write("-" * 72 + "\n")
        for index, label in enumerate(expansion.labels):
            handle.write(f"{index:2d}: {label}\n")

    np.savez_compressed(
        args.output_dir / "demo_dimer_duhamel_transport.npz",
        omega1_values=omega1_values,
        omega3_values=omega3_values,
        target_lam=target_lam,
        reference_model_signal=reference_model_signal,
        target_model_signal=target_model_signal,
        reference_pathway_signal=reference_pathway_signal,
        exact_target_pathway_signal=exact_target_pathway_signal,
        exact_projected_transport_signal=exact_projected_transport_signal,
        reconstructed_first=reconstructed_first,
        reconstructed_second=reconstructed_second,
        T_exact=T_exact,
        T_target=T_target,
        labels=np.asarray(expansion.labels, dtype=str),
        scan=scan,
        first_errors=first_errors,
        second_errors=second_errors,
    )

    print("Dimer Duhamel/transport proof of concept")
    print("=" * 72)
    print(f"target lambda: {np.array2string(target_lam, precision=8)}")
    print(f"physical displacements: {parameter_map.physical_displacements(target_lam)}")
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
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
