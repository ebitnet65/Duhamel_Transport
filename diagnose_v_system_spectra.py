#!/usr/bin/env python3
"""
diagnose_v_system_spectra.py

Check the expected V-system 2D spectral structure.

The default transition dipole couples |g> only to the bright state |B>. The
dark state |D> gains oscillator strength only through coherent B/D exchange.
This script compares:

1. no B/D mixing,
2. coherent B/D mixing,
3. coherent mixing plus detailed-balance incoherent transfer.

It reports peak-window amplitudes around BB, BD, DB, and DD coordinates and
saves real-part and magnitude panels for visual inspection.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os

import numpy as np

import Liouville_transport_general as general
from fit_v_system_transport_signal import commutator_signal_grid
from test_v_system_general import v_system_model, v_system_parameter_map

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")
import matplotlib.pyplot as plt


def parameter_map_for_case(*, J_bd: float, k_bd: float) -> general.ParameterMap:
    base = dict(v_system_parameter_map().reference_params)
    base["J_bd"] = J_bd
    base["k_bd"] = k_bd
    base["mu_gd"] = 0.0
    return general.ParameterMap(reference_params=base, controls=())


def window_peak(
    signal_grid: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    *,
    omega1_center: float,
    omega3_center: float,
    half_width: float,
) -> float:
    mask1 = np.abs(omega1_values - omega1_center) <= half_width
    mask3 = np.abs(omega3_values - omega3_center) <= half_width
    if not np.any(mask1) or not np.any(mask3):
        raise ValueError("Peak window contains no grid points")
    return float(np.max(np.abs(signal_grid[np.ix_(mask3, mask1)])))


def plot_case_grid(
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    component: str,
    output_file: Path,
) -> None:
    if component == "real":
        values = {name: np.real(signal) for name, signal in signals.items()}
        cmap = "RdBu_r"
        color_label = r"$\mathrm{Re}\,S$"
        vmax = max(float(np.max(np.abs(value))) for value in values.values())
        vmin = -vmax
    elif component == "abs":
        values = {name: np.abs(signal) for name, signal in signals.items()}
        cmap = "magma"
        color_label = r"$|S|$"
        vmin = 0.0
        vmax = max(float(np.max(value)) for value in values.values())
    else:
        raise ValueError("component must be 'real' or 'abs'")

    fig, axes = plt.subplots(1, len(values), figsize=(5.0 * len(values), 4.4), sharex=True, sharey=True)
    if len(values) == 1:
        axes = [axes]
    image = None
    for ax, (name, value) in zip(axes, values.items(), strict=True):
        image = ax.imshow(
            value,
            origin="lower",
            extent=(
                float(omega1_values[0]),
                float(omega1_values[-1]),
                float(omega3_values[0]),
                float(omega3_values[-1]),
            ),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(1.0, color="white", linewidth=0.8, alpha=0.55)
        ax.axvline(1.18, color="white", linewidth=0.8, alpha=0.55)
        ax.axhline(1.0, color="white", linewidth=0.8, alpha=0.55)
        ax.axhline(1.18, color="white", linewidth=0.8, alpha=0.55)
        ax.set_title(name)
        ax.set_xlabel(r"$\omega_1$")
    axes[0].set_ylabel(r"$\omega_3$")
    fig.colorbar(image, ax=axes, label=color_label, shrink=0.86)
    fig.tight_layout()
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_difference_grid(
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    reference_name: str,
    component: str,
    output_file: Path,
) -> None:
    reference = signals[reference_name]
    differences = {
        name: signal - reference
        for name, signal in signals.items()
        if name != reference_name
    }
    if component == "real":
        values = {name: np.real(signal) for name, signal in differences.items()}
        cmap = "RdBu_r"
        color_label = r"$\Delta\mathrm{Re}\,S$"
        vmax = max(float(np.max(np.abs(value))) for value in values.values())
        vmin = -vmax
    elif component == "abs":
        values = {name: np.abs(signal) for name, signal in differences.items()}
        cmap = "magma"
        color_label = r"$|\Delta S|$"
        vmin = 0.0
        vmax = max(float(np.max(value)) for value in values.values())
    else:
        raise ValueError("component must be 'real' or 'abs'")

    fig, axes = plt.subplots(1, len(values), figsize=(5.0 * len(values), 4.4), sharex=True, sharey=True)
    if len(values) == 1:
        axes = [axes]
    image = None
    for ax, (name, value) in zip(axes, values.items(), strict=True):
        image = ax.imshow(
            value,
            origin="lower",
            extent=(
                float(omega1_values[0]),
                float(omega1_values[-1]),
                float(omega3_values[0]),
                float(omega3_values[-1]),
            ),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(1.0, color="white", linewidth=0.8, alpha=0.55)
        ax.axvline(1.18, color="white", linewidth=0.8, alpha=0.55)
        ax.axhline(1.0, color="white", linewidth=0.8, alpha=0.55)
        ax.axhline(1.18, color="white", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{name} - {reference_name}")
        ax.set_xlabel(r"$\omega_1$")
    axes[0].set_ylabel(r"$\omega_3$")
    fig.colorbar(image, ax=axes, label=color_label, shrink=0.86)
    fig.tight_layout()
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose V-system 2D spectral features.")
    parser.add_argument("--n-omega", type=int, default=80)
    parser.add_argument("--omega-min", type=float, default=0.82)
    parser.add_argument("--omega-max", type=float, default=1.34)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2e-3)
    parser.add_argument("--J-bd", type=float, default=0.045)
    parser.add_argument("--k-bd", type=float, default=0.018)
    parser.add_argument("--peak-half-width", type=float, default=0.035)
    parser.add_argument("--output-dir", type=Path, default=Path("v_system_spectra_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = v_system_model()
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    cases = {
        "no mixing": parameter_map_for_case(J_bd=0.0, k_bd=0.0),
        f"coherent J={args.J_bd:g}": parameter_map_for_case(J_bd=args.J_bd, k_bd=0.0),
        f"J={args.J_bd:g}, k={args.k_bd:g}": parameter_map_for_case(
            J_bd=args.J_bd,
            k_bd=args.k_bd,
        ),
    }

    signals = {}
    peak_summary = {}
    reference_peaks = None
    for name, pmap in cases.items():
        flat_signal = commutator_signal_grid(
            model,
            pmap,
            pmap.zero(),
            omega1_values,
            omega3_values,
            tau2=args.tau2,
            eta=args.eta,
        )
        signal_grid = flat_signal.reshape(len(omega3_values), len(omega1_values))
        signals[name] = signal_grid
        params = pmap.params()
        omega_b = params["omega_b"]
        omega_d = params["omega_d"]
        peaks = {
            "BB": window_peak(
                signal_grid,
                omega1_values,
                omega3_values,
                omega1_center=omega_b,
                omega3_center=omega_b,
                half_width=args.peak_half_width,
            ),
            "BD": window_peak(
                signal_grid,
                omega1_values,
                omega3_values,
                omega1_center=omega_b,
                omega3_center=omega_d,
                half_width=args.peak_half_width,
            ),
            "DB": window_peak(
                signal_grid,
                omega1_values,
                omega3_values,
                omega1_center=omega_d,
                omega3_center=omega_b,
                half_width=args.peak_half_width,
            ),
            "DD": window_peak(
                signal_grid,
                omega1_values,
                omega3_values,
                omega1_center=omega_d,
                omega3_center=omega_d,
                half_width=args.peak_half_width,
            ),
        }
        peaks["offdiag_asymmetry"] = (peaks["BD"] - peaks["DB"]) / max(
            peaks["BD"] + peaks["DB"],
            1e-14,
        )
        if reference_peaks is None:
            reference_peaks = dict(peaks)
        for key in ("BB", "BD", "DB", "DD"):
            peaks[f"delta_{key}"] = peaks[key] - reference_peaks[key]
        peaks["delta_offdiag_asymmetry"] = (
            (peaks["delta_BD"] - peaks["delta_DB"])
            / max(abs(peaks["delta_BD"]) + abs(peaks["delta_DB"]), 1e-14)
        )
        peak_summary[name] = peaks

    plot_case_grid(
        omega1_values,
        omega3_values,
        signals,
        component="real",
        output_file=args.output_dir / "v_system_cases_real.png",
    )
    plot_case_grid(
        omega1_values,
        omega3_values,
        signals,
        component="abs",
        output_file=args.output_dir / "v_system_cases_abs.png",
    )
    plot_difference_grid(
        omega1_values,
        omega3_values,
        signals,
        reference_name="no mixing",
        component="real",
        output_file=args.output_dir / "v_system_cases_delta_real.png",
    )
    plot_difference_grid(
        omega1_values,
        omega3_values,
        signals,
        reference_name="no mixing",
        component="abs",
        output_file=args.output_dir / "v_system_cases_delta_abs.png",
    )

    summary_file = args.output_dir / "v_system_spectra_summary.txt"
    with summary_file.open("w", encoding="utf-8") as handle:
        handle.write("V-system spectra diagnostics\n")
        handle.write("=" * 72 + "\n\n")
        handle.write("Dipole convention: mu_gd = 0, mu_gb = 1\n")
        handle.write(f"peak half width: {args.peak_half_width:.8f}\n\n")
        for name, peaks in peak_summary.items():
            handle.write(f"{name}\n")
            handle.write("-" * 72 + "\n")
            for key in ("BB", "BD", "DB", "DD", "offdiag_asymmetry"):
                handle.write(f"{key:18s} {peaks[key]:.12e}\n")
            handle.write("\nrelative to no mixing\n")
            for key in ("delta_BB", "delta_BD", "delta_DB", "delta_DD", "delta_offdiag_asymmetry"):
                handle.write(f"{key:18s} {peaks[key]:.12e}\n")
            handle.write("\n")

    print("V-system spectra diagnostics")
    print("=" * 72)
    for name, peaks in peak_summary.items():
        print(
            f"{name:18s} "
            f"BB={peaks['BB']:.4e} BD={peaks['BD']:.4e} "
            f"DB={peaks['DB']:.4e} DD={peaks['DD']:.4e} "
            f"asym={peaks['offdiag_asymmetry']:.4e} "
            f"dBD={peaks['delta_BD']:.4e} dDB={peaks['delta_DB']:.4e} "
            f"dDD={peaks['delta_DD']:.4e}"
        )
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
