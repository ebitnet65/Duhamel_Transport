#!/usr/bin/env python3
"""
diagnose_exciton_dimer_spectra.py

Site-basis exciton dimer sanity check.

The one-exciton basis is

    |0> = |00>,  |1> = |10>,  |2> = |01>.

The transition dipole couples the ground state to both site excitations,

    mu = mu_site (|10><00| + |01><00| + h.c.).

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
from test_v_system_general import v_system_hamiltonian, v_system_model, v_system_parameter_map

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")


def parameter_map_for_dimer_case(*, J_bd: float, k_bd: float, mu_site: float) -> general.ParameterMap:
    base = dict(v_system_parameter_map().reference_params)
    base["J_bd"] = J_bd
    base["k_bd"] = k_bd
    base["mu_gb"] = mu_site
    base["mu_gd"] = mu_site
    return general.ParameterMap(reference_params=base, controls=())


def exciton_summary(params: general.Params) -> list[dict[str, float]]:
    """Return one-exciton energies and oscillator strengths for site dipoles."""

    hamiltonian = v_system_hamiltonian(params)
    one_exciton_hamiltonian = hamiltonian[1:3, 1:3]
    energies, vectors = np.linalg.eigh(one_exciton_hamiltonian)
    site_dipole = np.array([params["mu_gb"], params["mu_gd"]], dtype=complex)
    out = []
    for index, energy in enumerate(energies):
        vector = np.real_if_close(vectors[:, index])
        amplitude = np.vdot(site_dipole, vector)
        out.append(
            {
                "energy": float(energy),
                "site10_coeff": float(vector[0]),
                "site01_coeff": float(vector[1]),
                "oscillator_strength": float(abs(amplitude) ** 2),
            }
        )
    return out


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
    parser.add_argument("--J-bd", type=float, default=0.045)
    parser.add_argument("--k-bd", type=float, default=0.018)
    parser.add_argument("--mu-site", type=float, default=1.0)
    parser.add_argument("--peak-half-width", type=float, default=0.035)
    parser.add_argument("--output-dir", type=Path, default=Path("exciton_dimer_spectra_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = v_system_model()
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    cases = {
        "J=0": parameter_map_for_dimer_case(
            J_bd=0.0,
            k_bd=0.0,
            mu_site=args.mu_site,
        ),
        f"J=+{args.J_bd:g}": parameter_map_for_dimer_case(
            J_bd=abs(args.J_bd),
            k_bd=0.0,
            mu_site=args.mu_site,
        ),
        f"J=-{args.J_bd:g}": parameter_map_for_dimer_case(
            J_bd=-abs(args.J_bd),
            k_bd=0.0,
            mu_site=args.mu_site,
        ),
        f"J=+{args.J_bd:g}, k={args.k_bd:g}": parameter_map_for_dimer_case(
            J_bd=abs(args.J_bd),
            k_bd=args.k_bd,
            mu_site=args.mu_site,
        ),
        f"J=-{args.J_bd:g}, k={args.k_bd:g}": parameter_map_for_dimer_case(
            J_bd=-abs(args.J_bd),
            k_bd=args.k_bd,
            mu_site=args.mu_site,
        ),
    }

    signals = {}
    peak_summaries = {}
    exciton_summaries = {}
    for name, pmap in cases.items():
        params = pmap.params()
        excitons = exciton_summary(params)
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
        handle.write("Dipole convention: mu_gb = mu_gd = mu_site\n")
        handle.write("Hamiltonian convention: +J_bd (|10><01| + |01><10|)\n")
        handle.write(
            "For equal site energies and parallel site dipoles, this convention "
            "makes positive J brighten the upper symmetric exciton.\n\n"
        )
        for name in cases:
            handle.write(f"{name}\n")
            handle.write("-" * 72 + "\n")
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
