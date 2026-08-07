#!/usr/bin/env python3
"""Plot and animate rephasing spectra along iterative SAKE trajectories."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import Liouville_transport_general as general
from benchmark_sake_accuracy_timing import clean_dimer_parameter_map
from exciton_dimer_model import exciton_dimer_model
from fit_v_system_transport_signal import trace_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orange-run", type=Path, required=True)
    parser.add_argument("--green-run", type=Path, required=True)
    parser.add_argument("--mixing-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omega0", type=float, default=2.0)
    parser.add_argument("--delta-omega", type=float, default=0.18)
    parser.add_argument("--n-omega", type=int, default=64, help="Dense spectral grid; use at least 40 for figures.")
    parser.add_argument("--omega-min", type=float, default=1.7)
    parser.add_argument("--omega-max", type=float, default=2.3)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3e-3)
    parser.add_argument("--movie-frames", type=int, default=24)
    return parser.parse_args()


def model_args(args: argparse.Namespace, reference: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        omega0=args.omega0,
        delta_omega=args.delta_omega,
        delta_J=None,
        delta_k=None,
        reference_J=float(reference[0]),
        reference_k=float(reference[1]),
        clean_gamma=0.01,
        clean_gamma_phi=0.01,
        clean_mu_esa_scale=1.0,
    )


def trajectory(path: Path) -> np.ndarray:
    with (path / "iterations.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    references = np.array([
        [float(row["reference_J"]), float(row["reference_k"])] for row in rows
    ])
    final = np.array([float(rows[-1]["trial_J"]), float(rows[-1]["trial_k"])])
    return np.vstack([references, final])


def interpolate_trajectory(points: np.ndarray, count: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if arc[-1] == 0.0:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, arc[-1], count)
    return np.column_stack([
        np.interp(samples, arc, points[:, coordinate]) for coordinate in range(2)
    ])


def retained_pathways(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    frequencies: np.ndarray,
    args: argparse.Namespace,
    sector: str = "RP",
) -> list[general.CanonicalPathway]:
    pathways, _ = general.independent_canonical_pathways(
        model,
        parameter_map,
        frequencies,
        general.canonical_pathways((sector,)),
        stage="after_G3",
        tau2=args.tau2,
        eta=args.eta,
        rtol=1e-9,
    )
    return pathways


def fast_rephasing_spectrum(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    pathways: list[general.CanonicalPathway],
    frequencies: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    params = parameter_map.params(parameter_map.zero())
    L = model.L(params)
    rho = model.rho0(params)
    mu = model.mu(params)
    mu_components = model.mu_components(params)
    U2 = model.U(L, args.tau2)
    G1 = [model.G(L, omega, args.eta) for omega in frequencies]
    G3 = [model.G(L, -omega, args.eta) for omega in frequencies]
    pathway_spectra = []
    for item in pathways:
        J1 = general.interaction_super(*item.pathway[0], mu=mu, dipole_components=mu_components)
        J2 = general.interaction_super(*item.pathway[1], mu=mu, dipole_components=mu_components)
        J3 = general.interaction_super(*item.pathway[2], mu=mu, dipole_components=mu_components)
        after_J3 = [J3 @ U2 @ J2 @ G @ J1 @ rho for G in G1]
        values = np.empty((len(frequencies), len(frequencies)), dtype=complex)
        for i3, propagator in enumerate(G3):
            for i1, state in enumerate(after_J3):
                values[i3, i1] = trace_signal(model, params, propagator @ state)
        pathway_spectra.append(values)
    pathway_array = np.asarray(pathway_spectra)
    return np.sum(pathway_array, axis=0), pathway_array


def spectra_along(
    points: np.ndarray,
    model: general.LiouvilleModel,
    pathways: list[general.CanonicalPathway],
    frequencies: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    spectra = []
    for point in points:
        parameter_map = clean_dimer_parameter_map(model_args(args, point))
        spectrum, _ = fast_rephasing_spectrum(model, parameter_map, pathways, frequencies, args)
        spectra.append(spectrum)
    return np.asarray(spectra)


def plot_spectrum(
    axis: plt.Axes,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    title: str,
) -> None:
    real_spectrum = np.real(spectrum)
    scale = max(float(np.max(np.abs(real_spectrum))), 1e-14)
    normalized = real_spectrum / scale
    levels = np.linspace(-1.0, 1.0, 25)
    axis.contourf(frequencies, frequencies, normalized, levels=levels, cmap="RdBu_r", extend="both")
    axis.contour(
        frequencies,
        frequencies,
        normalized,
        levels=np.linspace(-1.0, 1.0, 9),
        colors="0.25",
        linewidths=0.25,
        alpha=0.55,
    )
    axis.plot(frequencies, frequencies, color="0.35", linewidth=0.5, linestyle=":")
    axis.set_title(title, fontsize=8)
    axis.set_aspect("equal")
    axis.tick_params(direction="in", top=True, right=True, labelsize=7)


def selected_indices(count: int) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, count - 1, 4)).astype(int))


def static_spectral_series(
    frequencies: np.ndarray,
    orange_points: np.ndarray,
    orange_spectra: np.ndarray,
    green_points: np.ndarray,
    green_spectra: np.ndarray,
    args: argparse.Namespace,
) -> None:
    orange_indices = selected_indices(len(orange_points))
    green_indices = selected_indices(len(green_points))
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), constrained_layout=True)
    orange_norm0 = np.linalg.norm(orange_spectra[0])
    green_norm0 = np.linalg.norm(green_spectra[0])
    for column, index in enumerate(orange_indices):
        point = orange_points[index] / args.omega0
        plot_spectrum(
            axes[0, column], frequencies, orange_spectra[index],
            rf"$({point[0]:.3f},{point[1]:.3f})$" + "\n" + rf"$A/A_0={np.linalg.norm(orange_spectra[index])/orange_norm0:.2f}$",
        )
    for column, index in enumerate(green_indices):
        point = green_points[index] / args.omega0
        plot_spectrum(
            axes[1, column], frequencies, green_spectra[index],
            rf"$({point[0]:.3f},{point[1]:.3f})$" + "\n" + rf"$A/A_0={np.linalg.norm(green_spectra[index])/green_norm0:.2f}$",
        )
    axes[0, 0].set_ylabel(r"Orange: $\omega_3$ (eV)")
    axes[1, 0].set_ylabel(r"Green: $\omega_3$ (eV)")
    for axis in axes[1]:
        axis.set_xlabel(r"$\omega_1$ (eV)")
    scalar = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1.0, 1.0))
    scalar.set_array([])
    fig.colorbar(scalar, ax=axes, shrink=0.86, pad=0.015, label=r"$\mathrm{Re}\,S_{\rm RP}/\max|\mathrm{Re}\,S_{\rm RP}|$")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"rephasing_spectral_evolution.{suffix}", dpi=400)
    plt.close(fig)


def spectral_movie(
    frequencies: np.ndarray,
    orange_points: np.ndarray,
    orange_spectra: np.ndarray,
    green_points: np.ndarray,
    green_spectra: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    orange_norm0 = np.linalg.norm(orange_spectra[0])
    green_norm0 = np.linalg.norm(green_spectra[0])

    def draw(frame: int):
        for axis in axes:
            axis.clear()
        orange = orange_points[frame] / args.omega0
        green = green_points[frame] / args.omega0
        plot_spectrum(
            axes[0], frequencies, orange_spectra[frame],
            rf"Orange: $({orange[0]:.3f},{orange[1]:.3f})$; $A/A_0={np.linalg.norm(orange_spectra[frame])/orange_norm0:.2f}$",
        )
        plot_spectrum(
            axes[1], frequencies, green_spectra[frame],
            rf"Green: $({green[0]:.3f},{green[1]:.3f})$; $A/A_0={np.linalg.norm(green_spectra[frame])/green_norm0:.2f}$",
        )
        axes[0].set_ylabel(r"$\omega_3$ (eV)")
        for axis in axes:
            axis.set_xlabel(r"$\omega_1$ (eV)")
        return axes

    movie = animation.FuncAnimation(fig, draw, frames=len(orange_points), interval=250, blit=False)
    movie.save(args.output_dir / "rephasing_trajectory_evolution.mp4", writer=animation.FFMpegWriter(fps=4, bitrate=2400))
    movie.save(args.output_dir / "rephasing_trajectory_evolution.gif", writer=animation.PillowWriter(fps=4))
    plt.close(fig)


def pathway_mixing(
    paths: list[Path],
    labels: list[str],
    pathway_labels: dict[str, list[str]],
    args: argparse.Namespace,
) -> None:
    matrices = []
    for path in paths:
        with np.load(path / "accumulated_transport.npz") as archive:
            full = np.asarray(archive["accumulated_transport"])
            matrices.append({"RP": full[:3, :3], "NRP": full[3:6, 3:6]})
    maximum = max(
        float(np.max(np.abs(matrix[sector])))
        for matrix in matrices for sector in ("RP", "NRP")
    )
    light_blue = LinearSegmentedColormap.from_list(
        "sake_light_blue", ["#ffffff", "#e7f1f8", "#bdd7ea", "#79add2"]
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    report_rows = []
    for sector_index, sector in enumerate(("RP", "NRP")):
        for target_index, (matrix_set, label) in enumerate(zip(matrices, labels, strict=True)):
            axis = axes[sector_index, target_index]
            matrix = matrix_set[sector]
            image = axis.imshow(np.abs(matrix), origin="lower", cmap=light_blue, vmin=0.0, vmax=maximum)
            if sector_index == 0:
                axis.set_title(label)
            axis.set_xticks(range(3), ["GSB", "SE", "ESA"])
            axis.set_yticks(range(3), ["GSB", "SE", "ESA"])
            if sector_index == 1:
                axis.set_xlabel("Reference pathway")
            if target_index == 0:
                axis.set_ylabel(f"{sector} target pathway")
            for row in range(3):
                for column in range(3):
                    magnitude = abs(matrix[row, column])
                    phase = np.degrees(np.angle(matrix[row, column]))
                    axis.text(column, row, f"{magnitude:.2f}", ha="center", va="center", fontsize=7, color="black")
                    report_rows.append({
                        "sector": sector,
                        "target": label,
                        "target_pathway": pathway_labels[sector][row],
                        "reference_pathway": pathway_labels[sector][column],
                        "real": float(matrix[row, column].real),
                        "imag": float(matrix[row, column].imag),
                        "magnitude": float(magnitude),
                        "phase_degrees": float(phase),
                    })
    fig.colorbar(image, ax=axes, shrink=0.84, pad=0.02, label=r"$|T_{ab}|$")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"pathway_mixing_rp_nrp.{suffix}", dpi=400)
    plt.close(fig)
    with (args.output_dir / "pathway_mixing_rp_nrp.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    model = exciton_dimer_model()
    zero_map = clean_dimer_parameter_map(model_args(args, np.zeros(2)))
    pathways = retained_pathways(model, zero_map, frequencies, args, "RP")
    nrp_pathways = retained_pathways(model, zero_map, frequencies, args, "NRP")

    orange_accepted = trajectory(args.orange_run)
    green_accepted = trajectory(args.green_run)
    orange_movie = interpolate_trajectory(orange_accepted, args.movie_frames)
    green_movie = interpolate_trajectory(green_accepted, args.movie_frames)
    orange_movie_spectra = spectra_along(orange_movie, model, pathways, frequencies, args)
    green_movie_spectra = spectra_along(green_movie, model, pathways, frequencies, args)
    orange_accepted_spectra = spectra_along(orange_accepted, model, pathways, frequencies, args)
    green_accepted_spectra = spectra_along(green_accepted, model, pathways, frequencies, args)
    np.savez_compressed(
        args.output_dir / "rephasing_evolution_data.npz",
        frequencies=frequencies,
        orange_points=orange_movie,
        orange_spectra=orange_movie_spectra,
        green_points=green_movie,
        green_spectra=green_movie_spectra,
    )
    static_spectral_series(
        frequencies,
        orange_accepted,
        orange_accepted_spectra,
        green_accepted,
        green_accepted_spectra,
        args,
    )
    spectral_movie(
        frequencies,
        orange_movie,
        orange_movie_spectra,
        green_movie,
        green_movie_spectra,
        args,
    )
    pathway_mixing(
        args.mixing_runs,
        ["T1: (-0.050, 0.006)", "T2: (0.040, 0.004)", "T3: (-0.030, 0.010)"],
        {
            "RP": [item.label for item in pathways],
            "NRP": [item.label for item in nrp_pathways],
        },
        args,
    )
    (args.output_dir / "pathway_labels_rp_nrp.txt").write_text(
        "RP\n" + "\n".join(f"P{index + 1}: {item.label}" for index, item in enumerate(pathways))
        + "\n\nNRP\n" + "\n".join(f"P{index + 1}: {item.label}" for index, item in enumerate(nrp_pathways)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
