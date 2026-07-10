#!/usr/bin/env python3
"""
run_liouville_transport.py

Driver for the Liouville-space pathway transport calculation.

Required files in the same directory:
    phase_matched_response.py
    Liouville_transport.py

Outputs:
    transport_output/liouville_transport_results.npz
    transport_output/transport_summary.txt
    transport_output/T_abs_final.png
    transport_output/T_real_final.png
    transport_output/T_imag_final.png
    transport_output/T_elements_abs.png
    transport_output/projection_residuals.png
    transport_output/condition_number.png
    
    Within this, we adopt the following notion
    RP: L-L+L+   = rephasing, ground state bleach
    RP: L-R+L+   = rephasing, stimulated emission
    
    
    
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

from Liouville_transport import (
    liouville_transport_sweep,
    print_liouville_transport_diagnostics,
    plot_liouville_transport_elements,
    plot_liouville_transport_heatmap,
)


def print_banner(args, omegas, lambda_values):
    """Print a summary of the transport calculation."""

    print("=" * 78)
    print("             Liouville Pathway Transport Analysis")
    print("      Geometric Reconstruction of Nonlinear Response")
    print("=" * 78)

    print("\nTheory")
    print("------")
    print("Canonical pathways are transported on the Liouville manifold")
    print("according to")
    print("")
    print("    |P_a(λ)> = Σ_b T_ab(λ) |P_b(0)>")
    print("")
    print("where T_ab is obtained by least-squares projection of the")
    print("transported pathway basis onto the canonical λ=0 basis.")
    print("")
    print(f"Transport stage        : after_U2")
    print(f"Initial state          : {args.initial}")
    print(f"Independent basis      : by_sector")
    print(f"Phase matching         : RP and NRP computed independently")
    print(f"Basis reduction        : enabled")
    print(f"Projection method      : SVD pseudoinverse")
    print(f"Quadrature weighting   : frequency weighted")

    print("\nModel")
    print("-----")
    print(f"ω₀                     = {args.w0:8.4f}")
    print(f"γφ                     = {args.gamma_phi:8.4f}")
    print(f"γ₁                     = {args.gamma1:8.4f}")
    print(f"τ₂                     = {args.tau2:8.4f}")
    print(f"η                      = {args.eta:8.2e}")

    print("\nParameter Scan")
    print("--------------")
    print(
        f"θ range                = "
        f"{lambda_values[0]:.3f} → {lambda_values[-1]:.3f}"
    )
    print(f"Number of θ values     = {len(lambda_values)}")
    print(
        f"Frequency window       = "
        f"{omegas[0]:.3f} → {omegas[-1]:.3f}"
    )
    print(f"Frequency samples      = {len(omegas)}")

    print("\nOutput")
    print("------")
    print("• Independent canonical pathway identification")
    print("• Liouville transport matrices T_ab(θ)")
    print("• Heat maps (|T|, Re T, Im T)")
    print("• Transport-element evolution")
    print("• Projection residual diagnostics")
    print("• Basis conditioning diagnostics")

    print("=" * 78)
    print()
    


def plot_projection_residuals(result: dict, output_file: Path) -> None:
    """Plot the pathway projection residuals versus lambda."""
    lam = result["lambda_values"]
    residuals = result["residuals"]
    labels = result["labels"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, label in enumerate(labels):
        ax.plot(
            lam,
            residuals[:, index],
            marker="o",
            markersize=3,
            label=label,
        )

    ax.set_xlabel(r"$\lambda\equiv\theta$")
    ax.set_ylabel("relative projection residual")
    ax.set_title("Closure of the canonical pathway subspace")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_condition_number(result: dict, output_file: Path) -> None:
    """Plot the condition number of the reference pathway basis."""
    lam = result["lambda_values"]
    condition_numbers = result["condition_numbers"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lam, condition_numbers, marker="o", markersize=3)
    ax.set_xlabel(r"$\lambda\equiv\theta$")
    ax.set_ylabel("condition number")
    ax.set_yscale("log")
    ax.set_title("Conditioning of the canonical Liouville-pathway basis")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(result: dict, output_file: Path) -> None:
    """Write a compact text summary of the transport calculation."""
    T0 = result["T"][0]
    Tfinal = result["T"][-1]
    identity = np.eye(T0.shape[0], dtype=complex)

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("Liouville-space pathway transport summary\n")
        handle.write("=" * 72 + "\n\n")

        handle.write("Canonical pathways\n")
        handle.write("-" * 72 + "\n")
        for index, label in enumerate(result["labels"]):
            handle.write(f"{index:2d}: {label}\n")

        handle.write("\nIdentity check\n")
        handle.write("-" * 72 + "\n")
        handle.write(f"||T(0)-I|| = {np.linalg.norm(T0 - identity):.12e}\n")

        handle.write("\nRanks\n")
        handle.write("-" * 72 + "\n")
        for lam, rank in zip(result["lambda_values"], result["ranks"]):
            handle.write(f"lambda={lam:.8f}  rank={rank:d}\n")

        handle.write("\nCondition numbers\n")
        handle.write("-" * 72 + "\n")
        for lam, cond in zip(
            result["lambda_values"],
            result["condition_numbers"],
        ):
            handle.write(f"lambda={lam:.8f}  cond={cond:.12e}\n")

        handle.write("\nMaximum residual at each lambda\n")
        handle.write("-" * 72 + "\n")
        for lam, residuals in zip(
            result["lambda_values"],
            result["residuals"],
        ):
            handle.write(
                f"lambda={lam:.8f}  "
                f"max_residual={np.max(residuals):.12e}\n"
            )

        handle.write("\nT(lambda_max)\n")
        handle.write("-" * 72 + "\n")
        handle.write(np.array2string(Tfinal, precision=8))
        handle.write("\n\n|T(lambda_max)|\n")
        handle.write("-" * 72 + "\n")
        handle.write(np.array2string(np.abs(Tfinal), precision=8))
        handle.write("\n")


def save_npz(result: dict, output_file: Path) -> None:
    """Save numerical arrays without serializing pathway tuple objects."""
    np.savez_compressed(
        output_file,
        lambda_values=result["lambda_values"],
        labels=np.asarray(result["labels"], dtype=str),
        T=result["T"],
        residuals=result["residuals"],
        ranks=result["ranks"],
        condition_numbers=result["condition_numbers"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Liouville-space pathway transport."
    )

    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--gamma-phi", type=float, default=0.1)
    parser.add_argument("--gamma1", type=float, default=0.02)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)

    parser.add_argument("--omega-half-width", type=float, default=0.75)
    parser.add_argument("--n-omega", type=int, default=1001)

    parser.add_argument("--lambda-max", type=float, default=np.pi / 2)
    parser.add_argument("--n-lambda", type=int, default=41)

    parser.add_argument(
        "--initial",
        choices=("fixed_steady0", "steady", "ground"),
        default="fixed_steady0",
    )

    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--rcond", type=float, default=1e-12)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transport_output"),
    )

    return parser.parse_args()


def analyze_two_pathway_transport(result):
    """
    Decompose each 2x2 transport matrix into:

        T = U H

    and test whether H is an isotropic scalar contraction.
    """
    lambdas = result["lambda_values"]

    singular_values = []
    anisotropy = []
    scalar_attenuation = []
    unitarity_error = []
    su2_matrices = []

    for T in result["T"]:
        if T.shape != (2, 2):
            raise ValueError("This diagnostic requires a 2x2 transport matrix.")

        # Polar decomposition T = U H
        U, H = linalg.polar(T)

        s = np.linalg.svd(T, compute_uv=False)
        singular_values.append(s)

        mean_s = np.mean(s)
        scalar_attenuation.append(mean_s)

        anisotropy.append(
            np.abs(s[0] - s[1]) / max(mean_s, 1e-14)
        )

        # Remove the overall U(1) phase to obtain det = 1.
        det_u = np.linalg.det(U)
        U_su2 = U * np.exp(-0.5j * np.angle(det_u))
        su2_matrices.append(U_su2)

        unitarity_error.append(
            np.linalg.norm(U_su2.conj().T @ U_su2 - np.eye(2))
        )

    return {
        "lambda_values": lambdas,
        "singular_values": np.asarray(singular_values),
        "scalar_attenuation": np.asarray(scalar_attenuation),
        "anisotropy": np.asarray(anisotropy),
        "unitarity_error": np.asarray(unitarity_error),
        "U_su2": np.asarray(su2_matrices),
    }
    




def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    omegas = np.linspace(
        args.w0 - args.omega_half_width,
        args.w0 + args.omega_half_width,
        args.n_omega,
    )

    lambda_values = np.linspace(
        0.0,
        args.lambda_max,
        args.n_lambda,
    )
    
    print_banner(args,omegas,lambda_values)

    sector_results = {}

    for sector in ("RP", "NRP"):
        print("\n" + "=" * 80)
        print(f"Computing {sector} pathway transport")
        print("=" * 80)

        result = liouville_transport_sweep(
            lambda_values=lambda_values,
            omegas=omegas,
            sectors=(sector,),
            stage="after_U2",
            independence_mode="by_sector",
            select_independent=True,
            independence_rtol=1e-10,
            cutoff=args.cutoff,
            tau2=args.tau2,
            w0=args.w0,
            gamma_phi=args.gamma_phi,
            gamma1=args.gamma1,
            eta=args.eta,
            initial=args.initial,
            rcond=args.rcond,
            verbose=True,
        )

        sector_results[sector] = result

        sector_dir = output_dir / sector
        sector_dir.mkdir(parents=True, exist_ok=True)

        print_liouville_transport_diagnostics(result)

        save_npz(
            result,
            sector_dir / f"{sector}_liouville_transport_results.npz",
        )

        write_summary(
            result,
            sector_dir / f"{sector}_transport_summary.txt",
        )

        plot_liouville_transport_heatmap(
            result,
            lambda_index=-1,
            component="abs",
            savefile=sector_dir / f"{sector}_T_abs_final.png",
            show=False,
        )

        plot_liouville_transport_heatmap(
            result,
            lambda_index=-1,
            component="real",
            savefile=sector_dir / f"{sector}_T_real_final.png",
            show=False,
        )

        plot_liouville_transport_heatmap(
            result,
            lambda_index=-1,
            component="imag",
            savefile=sector_dir / f"{sector}_T_imag_final.png",
            show=False,
        )

        plot_liouville_transport_elements(
            result,
            component="abs",
            threshold=1e-3,
            savefile=sector_dir / f"{sector}_T_elements_abs.png",
            show=False,
        )

        plot_projection_residuals(
            result,
            sector_dir / f"{sector}_projection_residuals.png",
        )

        plot_condition_number(
            result,
            sector_dir / f"{sector}_condition_number.png",
        )

        print(f"{sector} results written to: {sector_dir.resolve()}")
        
    rp_analysis = analyze_two_pathway_transport(sector_results["RP"])

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        rp_analysis["lambda_values"],
        rp_analysis["singular_values"][:, 0],
        label=r"$s_1$",
    )
    plt.plot(
        rp_analysis["lambda_values"],
        rp_analysis["singular_values"][:, 1],
        linestyle="--",
        label=r"$s_2$",
    )
    plt.xlabel(r"$\lambda\equiv\theta$")
    plt.ylabel("singular value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()