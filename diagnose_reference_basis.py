#!/usr/bin/env python3
"""
diagnose_reference_basis.py

Diagnose linear dependence in the canonical Liouville-pathway basis.

Expected files in the same directory:
    phase_matched_response.py
    Liouville_transport.py

The script evaluates RP, NRP, and combined RP+NRP pathway bases and reports:
    - pathway norms
    - singular values
    - normalized Gram matrix
    - pairwise overlaps
    - effective rank
    - condition number

It also saves:
    basis_diagnostics/
        RP_summary.txt
        NRP_summary.txt
        RP_NRP_summary.txt
        RP_gram_abs.png
        NRP_gram_abs.png
        RP_NRP_gram_abs.png
        RP_singular_values.png
        NRP_singular_values.png
        RP_NRP_singular_values.png
        basis_diagnostics.npz
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from Liouville_transport import (
    canonical_pathway_list,
    liouville_pathway_basis_matrix,
)


def normalize_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=0)
    safe = np.maximum(norms, 1e-15)
    return matrix / safe[None, :], norms


def diagnose_basis(
    omegas: np.ndarray,
    canonical_pathways: list[dict],
    *,
    tau2: float,
    w0: float,
    gamma_phi: float,
    gamma1: float,
    eta: float,
    initial: str,
) -> dict:
    P0 = liouville_pathway_basis_matrix(
        theta=0.0,
        omegas=omegas,
        canonical_pathways=canonical_pathways,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        initial=initial,
    )

    Pn, norms = normalize_columns(P0)
    _, singular_values, Vh = np.linalg.svd(Pn, full_matrices=False)

    if singular_values.size == 0 or singular_values[0] == 0:
        relative_singular_values = singular_values.copy()
    else:
        relative_singular_values = singular_values / singular_values[0]

    gram = Pn.conj().T @ Pn

    if singular_values.size == 0 or singular_values[-1] == 0:
        condition_number = np.inf
    else:
        condition_number = singular_values[0] / singular_values[-1]

    effective_ranks = {
        tol: int(np.sum(relative_singular_values > tol))
        for tol in (1e-8, 1e-10, 1e-12, 1e-14)
    }

    return {
        "P0": P0,
        "Pn": Pn,
        "norms": norms,
        "gram": gram,
        "singular_values": singular_values,
        "relative_singular_values": relative_singular_values,
        "right_singular_vectors": Vh,
        "condition_number": condition_number,
        "effective_ranks": effective_ranks,
    }


def write_summary(
    output_file: Path,
    sector_label: str,
    labels: list[str],
    diagnostic: dict,
) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"Reference-basis diagnostic: {sector_label}\n")
        handle.write("=" * 78 + "\n\n")

        handle.write("Pathway norms\n")
        handle.write("-" * 78 + "\n")
        for label, norm in zip(labels, diagnostic["norms"]):
            handle.write(f"{label:32s} {norm:16.8e}\n")

        handle.write("\nSingular values of normalized basis\n")
        handle.write("-" * 78 + "\n")
        for index, (value, relative) in enumerate(
            zip(
                diagnostic["singular_values"],
                diagnostic["relative_singular_values"],
            )
        ):
            handle.write(
                f"{index:2d}: {value:16.8e}   relative={relative:16.8e}\n"
            )

        handle.write("\nCondition number\n")
        handle.write("-" * 78 + "\n")
        handle.write(f"{diagnostic['condition_number']:.12e}\n")

        handle.write("\nEffective rank\n")
        handle.write("-" * 78 + "\n")
        for tol, rank in diagnostic["effective_ranks"].items():
            handle.write(f"relative tolerance={tol:8.1e}  rank={rank}\n")

        handle.write("\nNormalized Gram matrix\n")
        handle.write("-" * 78 + "\n")
        handle.write(
            np.array2string(
                diagnostic["gram"],
                precision=6,
                suppress_small=True,
            )
        )
        handle.write("\n\nAbsolute overlaps\n")
        handle.write("-" * 78 + "\n")
        handle.write(
            np.array2string(
                np.abs(diagnostic["gram"]),
                precision=6,
                suppress_small=True,
            )
        )
        handle.write("\n\nRight singular vectors\n")
        handle.write("-" * 78 + "\n")
        handle.write(
            np.array2string(
                diagnostic["right_singular_vectors"],
                precision=6,
                suppress_small=True,
            )
        )
        handle.write("\n")


def plot_gram(
    gram: np.ndarray,
    labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.abs(gram), origin="lower", aspect="equal")
    fig.colorbar(image, ax=ax, label=r"$|\langle P_a|P_b\rangle|$")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_singular_values(
    relative_singular_values: np.ndarray,
    title: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    index = np.arange(1, len(relative_singular_values) + 1)
    ax.semilogy(index, relative_singular_values, marker="o")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"$\sigma_i/\sigma_{\max}$")
    ax.set_title(title)
    ax.set_xticks(index)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the canonical Liouville-pathway basis."
    )

    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--gamma-phi", type=float, default=0.1)
    parser.add_argument("--gamma1", type=float, default=0.02)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)

    parser.add_argument("--omega-half-width", type=float, default=0.75)
    parser.add_argument("--n-omega", type=int, default=1001)

    parser.add_argument(
        "--initial",
        choices=("fixed_steady0", "steady", "ground"),
        default="fixed_steady0",
    )

    parser.add_argument("--cutoff", type=float, default=1e-10)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("basis_diagnostics"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    omegas = np.linspace(
        args.w0 - args.omega_half_width,
        args.w0 + args.omega_half_width,
        args.n_omega,
    )

    sector_sets = [
        ("RP",),
        ("NRP",),
        ("RP", "NRP"),
    ]

    npz_data: dict[str, np.ndarray] = {}

    for sectors in sector_sets:
        sector_label = "_".join(sectors)
        print("\n" + "=" * 78)
        print("SECTORS:", sectors)

        canonical = canonical_pathway_list(
            omegas=omegas,
            sectors=sectors,
            cutoff=args.cutoff,
            tau2=args.tau2,
            w0=args.w0,
            gamma_phi=args.gamma_phi,
            gamma1=args.gamma1,
            eta=args.eta,
            initial=args.initial,
        )

        labels = [item["label"] for item in canonical]

        diagnostic = diagnose_basis(
            omegas,
            canonical,
            tau2=args.tau2,
            w0=args.w0,
            gamma_phi=args.gamma_phi,
            gamma1=args.gamma1,
            eta=args.eta,
            initial=args.initial,
        )

        print("Pathways:")
        for label, norm in zip(labels, diagnostic["norms"]):
            print(f"  {label:32s} ||P||={norm:.6e}")

        print("Relative singular values:")
        print(diagnostic["relative_singular_values"])

        print("Condition number:")
        print(diagnostic["condition_number"])

        print("Effective ranks:")
        print(diagnostic["effective_ranks"])

        print("Absolute Gram matrix:")
        print(np.abs(diagnostic["gram"]))

        write_summary(
            args.output_dir / f"{sector_label}_summary.txt",
            sector_label,
            labels,
            diagnostic,
        )

        plot_gram(
            diagnostic["gram"],
            labels,
            f"{sector_label} normalized pathway overlaps",
            args.output_dir / f"{sector_label}_gram_abs.png",
        )

        plot_singular_values(
            diagnostic["relative_singular_values"],
            f"{sector_label} normalized singular values",
            args.output_dir / f"{sector_label}_singular_values.png",
        )

        npz_data[f"{sector_label}_labels"] = np.asarray(labels, dtype=str)
        npz_data[f"{sector_label}_norms"] = diagnostic["norms"]
        npz_data[f"{sector_label}_gram"] = diagnostic["gram"]
        npz_data[f"{sector_label}_singular_values"] = diagnostic["singular_values"]
        npz_data[
            f"{sector_label}_relative_singular_values"
        ] = diagnostic["relative_singular_values"]
        npz_data[
            f"{sector_label}_right_singular_vectors"
        ] = diagnostic["right_singular_vectors"]

    np.savez_compressed(
        args.output_dir / "basis_diagnostics.npz",
        omegas=omegas,
        **npz_data,
    )

    print(f"\nDiagnostics written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
