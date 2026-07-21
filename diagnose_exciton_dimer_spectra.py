#!/usr/bin/env python3
"""
diagnose_exciton_dimer_spectra.py

Site-basis exciton dimer sanity check.

The one-exciton basis is

    |0> = |00>,  |1> = |10>,  |2> = |01>,  |3> = |11>.

The transition dipole couples the ground state to both site excitations,

    mu = mu_site (|10><00| + |01><00| + |11><10| + |11><01| + h.c.).

This differs from `diagnose_v_system_spectra.py`, which treats state 2 as a
nominally dark state with no direct transition dipole.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os

import numpy as np

import Liouville_transport_general as general
from diagnose_v_system_spectra import plot_case_grid, plot_difference_grid, window_peak
from fit_v_system_transport_signal import commutator_signal_grid
from exciton_dimer_model import (
    excited_state_absorption_frequencies,
    exciton_dimer_model,
    exciton_dimer_parameter_map,
    one_exciton_summary,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")


def parameter_map_for_dimer_case(
    *,
    J_12: float,
    k_12: float,
    mu_site: float,
    mu_esa_scale: float,
) -> general.ParameterMap:
    base = dict(exciton_dimer_parameter_map().reference_params)
    base["J_12"] = J_12
    base["k_12"] = k_12
    base["mu_1"] = mu_site
    base["mu_2"] = mu_site
    base["mu_esa_scale"] = mu_esa_scale
    return general.ParameterMap(reference_params=base, controls=())


def peak_summary_for_signal(
    signal_grid: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    centers: tuple[float, float],
    *,
    half_width: float,
) -> dict[str, float]:
    lower, upper = centers
    peaks = {
        "LL": window_peak(
            signal_grid,
            omega1_values,
            omega3_values,
            omega1_center=lower,
            omega3_center=lower,
            half_width=half_width,
        ),
        "LU": window_peak(
            signal_grid,
            omega1_values,
            omega3_values,
            omega1_center=lower,
            omega3_center=upper,
            half_width=half_width,
        ),
        "UL": window_peak(
            signal_grid,
            omega1_values,
            omega3_values,
            omega1_center=upper,
            omega3_center=lower,
            half_width=half_width,
        ),
        "UU": window_peak(
            signal_grid,
            omega1_values,
            omega3_values,
            omega1_center=upper,
            omega3_center=upper,
            half_width=half_width,
        ),
    }
    peaks["offdiag_asymmetry"] = (peaks["LU"] - peaks["UL"]) / max(
        peaks["LU"] + peaks["UL"],
        1e-14,
    )
    return peaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose site-basis exciton dimer spectra.")
    parser.add_argument("--n-omega", type=int, default=80)
    parser.add_argument("--omega-min", type=float, default=0.78)
    parser.add_argument("--omega-max", type=float, default=1.38)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2e-3)
    parser.add_argument("--J-12", type=float, default=0.045)
    parser.add_argument("--k-12", type=float, default=0.018)
    parser.add_argument("--mu-site", type=float, default=1.0)
    parser.add_argument("--esa-scale", type=float, default=1.0)
    parser.add_argument("--peak-half-width", type=float, default=0.035)
    parser.add_argument("--output-dir", type=Path, default=Path("exciton_dimer_spectra_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = exciton_dimer_model()
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    cases = {
        "J=0": parameter_map_for_dimer_case(
            J_12=0.0,
            k_12=0.0,
            mu_site=args.mu_site,
            mu_esa_scale=0.0,
        ),
        f"J=+{args.J_12:g}, ESA off": parameter_map_for_dimer_case(
            J_12=abs(args.J_12),
            k_12=0.0,
            mu_site=args.mu_site,
            mu_esa_scale=0.0,
        ),
        f"J=+{args.J_12:g}, ESA on": parameter_map_for_dimer_case(
            J_12=abs(args.J_12),
            k_12=0.0,
            mu_site=args.mu_site,
            mu_esa_scale=args.esa_scale,
        ),
        f"J=-{args.J_12:g}, ESA off": parameter_map_for_dimer_case(
            J_12=-abs(args.J_12),
            k_12=0.0,
            mu_site=args.mu_site,
            mu_esa_scale=0.0,
        ),
        f"J=-{args.J_12:g}, ESA on": parameter_map_for_dimer_case(
            J_12=-abs(args.J_12),
            k_12=0.0,
            mu_site=args.mu_site,
            mu_esa_scale=args.esa_scale,
        ),
        f"J=+{args.J_12:g}, ESA on, k={args.k_12:g}": parameter_map_for_dimer_case(
            J_12=abs(args.J_12),
            k_12=args.k_12,
            mu_site=args.mu_site,
            mu_esa_scale=args.esa_scale,
        ),
        f"J=-{args.J_12:g}, ESA on, k={args.k_12:g}": parameter_map_for_dimer_case(
            J_12=-abs(args.J_12),
            k_12=args.k_12,
            mu_site=args.mu_site,
            mu_esa_scale=args.esa_scale,
        ),
    }

    signals = {}
    peak_summaries = {}
    exciton_summaries = {}
    for name, pmap in cases.items():
        params = pmap.params()
        excitons = one_exciton_summary(params)
        exciton_summaries[name] = excitons
        centers = (excitons[0]["energy"], excitons[1]["energy"])
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
        peak_summaries[name] = peak_summary_for_signal(
            signal_grid,
            omega1_values,
            omega3_values,
            centers,
            half_width=args.peak_half_width,
        )

    plot_case_grid(
        omega1_values,
        omega3_values,
        signals,
        component="real",
        output_file=args.output_dir / "exciton_dimer_cases_real.png",
    )
    plot_case_grid(
        omega1_values,
        omega3_values,
        signals,
        component="abs",
        output_file=args.output_dir / "exciton_dimer_cases_abs.png",
    )
    plot_difference_grid(
        omega1_values,
        omega3_values,
        signals,
        reference_name="J=0",
        component="real",
        output_file=args.output_dir / "exciton_dimer_delta_real.png",
    )
    plot_difference_grid(
        omega1_values,
        omega3_values,
        signals,
        reference_name="J=0",
        component="abs",
        output_file=args.output_dir / "exciton_dimer_delta_abs.png",
    )

    summary_file = args.output_dir / "exciton_dimer_spectra_summary.txt"
    with summary_file.open("w", encoding="utf-8") as handle:
        handle.write("Exciton dimer spectra diagnostics\n")
        handle.write("=" * 72 + "\n\n")
        handle.write("Basis: |00>, |10>, |01>, |11>\n")
        handle.write("Dipole convention: mu_1 = mu_2 = mu_site\n")
        handle.write("ESA convention: mu_esa_scale multiplies |11><10| and |11><01| transitions\n")
        handle.write("Hamiltonian convention: +J_12 (|10><01| + |01><10|)\n")
        handle.write(
            "For equal site energies and parallel site dipoles, this convention "
            "makes positive J brighten the upper symmetric exciton.\n\n"
        )
        for name in cases:
            params = cases[name].params()
            esa_frequencies = excited_state_absorption_frequencies(params)
            handle.write(f"{name}\n")
            handle.write("-" * 72 + "\n")
            handle.write(
                "ESA frequencies from lower/upper one-exciton states: "
                f"{esa_frequencies[0]:.12e}, {esa_frequencies[1]:.12e}\n"
            )
            for index, exciton in enumerate(exciton_summaries[name]):
                handle.write(
                    f"exciton {index}: energy={exciton['energy']:.12e} "
                    f"c10={exciton['site10_coeff']:.8f} "
                    f"c01={exciton['site01_coeff']:.8f} "
                    f"osc={exciton['oscillator_strength']:.12e}\n"
                )
            for key in ("LL", "LU", "UL", "UU", "offdiag_asymmetry"):
                handle.write(f"{key:18s} {peak_summaries[name][key]:.12e}\n")
            handle.write("\n")

    print("Exciton dimer spectra diagnostics")
    print("=" * 72)
    for name in cases:
        excitons = exciton_summaries[name]
        lower = excitons[0]
        upper = excitons[1]
        peaks = peak_summaries[name]
        print(
            f"{name:18s} "
            f"E_low={lower['energy']:.5f} osc_low={lower['oscillator_strength']:.4f} "
            f"E_up={upper['energy']:.5f} osc_up={upper['oscillator_strength']:.4f} "
            f"LU={peaks['LU']:.3e} UL={peaks['UL']:.3e} asym={peaks['offdiag_asymmetry']:.3e}"
        )
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
