#!/usr/bin/env python3
"""
diagnose_transport_stages.py

Compare pathway-basis conditioning at successive points in the
third-order response sequence:

    stage 1: after the second interaction
        J2 G1 J1 rho0

    stage 2: after waiting-time propagation
        U2 J2 G1 J1 rho0

    stage 3: after the third interaction
        J3 U2 J2 G1 J1 rho0

    stage 4: after the final frequency-domain propagation
        G3 J3 U2 J2 G1 J1 rho0

The script reports singular values, effective rank, condition number,
and Gram matrices for RP, NRP, and combined RP+NRP bases.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from phase_matched_response import (
    G_freq,
    initial_state,
    interaction_super,
    liouvillian,
    omega1_sign_for_sector,
)

from Liouville_transport import canonical_pathway_list


STAGE_NAMES = {
    "after_J2": r"$J_2G_1J_1\rho_0$",
    "after_U2": r"$U_2J_2G_1J_1\rho_0$",
    "after_J3": r"$J_3U_2J_2G_1J_1\rho_0$",
    "after_G3": r"$G_3J_3U_2J_2G_1J_1\rho_0$",
}


def pathway_state_at_stage(
    *,
    theta: float,
    omega: float,
    item: dict,
    stage: str,
    tau2: float,
    w0: float,
    gamma_phi: float,
    gamma1: float,
    eta: float,
    initial: str,
) -> np.ndarray:
    sector = item["sector"]
    pathway = item["pathway"]

    L = liouvillian(
        w0=w0,
        gamma_phi=gamma_phi,
        theta=theta,
        gamma1=gamma1,
    )

    L_ref = liouvillian(
        w0=w0,
        gamma_phi=gamma_phi,
        theta=0.0,
        gamma1=gamma1,
    )

    rho0 = initial_state(
        L,
        mode=initial,
        L_ref=L_ref,
    )

    J1 = interaction_super(*pathway[0])
    J2 = interaction_super(*pathway[1])
    J3 = interaction_super(*pathway[2])

    s1 = omega1_sign_for_sector(sector)

    G1 = G_freq(L, s1 * omega, eta=eta)
    G3 = G_freq(L, omega, eta=eta)
    U2 = scipy.linalg.expm(L * tau2)

    state = G1 @ J1 @ rho0
    state = J2 @ state

    if stage == "after_J2":
        return state

    state = U2 @ state

    if stage == "after_U2":
        return state

    state = J3 @ state

    if stage == "after_J3":
        return state

    if stage == "after_G3":
        return G3 @ state

    raise ValueError(f"Unknown stage: {stage}")


def stage_basis_matrix(
    omegas: np.ndarray,
    canonical: list[dict],
    stage: str,
    *,
    tau2: float,
    w0: float,
    gamma_phi: float,
    gamma1: float,
    eta: float,
    initial: str,
) -> np.ndarray:
    domega = np.gradient(omegas)
    sqrt_weight = np.sqrt(domega)[:, None]

    columns = []

    for item in canonical:
        states = np.array([
            pathway_state_at_stage(
                theta=0.0,
                omega=omega,
                item=item,
                stage=stage,
                tau2=tau2,
                w0=w0,
                gamma_phi=gamma_phi,
                gamma1=gamma1,
                eta=eta,
                initial=initial,
            )
            for omega in omegas
        ])

        weighted = sqrt_weight * states
        columns.append(weighted.reshape(-1))

    return np.column_stack(columns)


def diagnose_matrix(matrix: np.ndarray) -> dict:
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.maximum(norms, 1e-15)[None, :]

    _, singular_values, Vh = np.linalg.svd(normalized, full_matrices=False)

    if singular_values.size == 0 or singular_values[0] == 0:
        relative = singular_values.copy()
    else:
        relative = singular_values / singular_values[0]

    gram = normalized.conj().T @ normalized

    if singular_values.size == 0 or singular_values[-1] == 0:
        condition_number = np.inf
    else:
        condition_number = singular_values[0] / singular_values[-1]

    ranks = {
        tol: int(np.sum(relative > tol))
        for tol in (1e-8, 1e-10, 1e-12, 1e-14)
    }

    return {
        "norms": norms,
        "gram": gram,
        "singular_values": singular_values,
        "relative_singular_values": relative,
        "condition_number": condition_number,
        "effective_ranks": ranks,
        "right_singular_vectors": Vh,
    }


def plot_condition_by_stage(
    stage_results: dict[str, dict],
    title: str,
    output_file: Path,
) -> None:
    stages = list(stage_results)
    values = [stage_results[stage]["condition_number"] for stage in stages]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(np.arange(len(stages)), values, marker="o")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels([STAGE_NAMES[stage] for stage in stages], rotation=20)
    ax.set_ylabel("condition number")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_gram(
    diagnostic: dict,
    labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        np.abs(diagnostic["gram"]),
        origin="lower",
        aspect="equal",
    )
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


def write_stage_summary(
    output_file: Path,
    sector_label: str,
    labels: list[str],
    stage_results: dict[str, dict],
) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"Pathway-basis conditioning by stage: {sector_label}\n")
        handle.write("=" * 80 + "\n\n")

        handle.write("Pathways\n")
        handle.write("-" * 80 + "\n")
        for index, label in enumerate(labels):
            handle.write(f"{index:2d}: {label}\n")

        for stage, diagnostic in stage_results.items():
            handle.write("\n" + STAGE_NAMES[stage] + "\n")
            handle.write("-" * 80 + "\n")
            handle.write(
                f"condition number = "
                f"{diagnostic['condition_number']:.12e}\n"
            )
            handle.write(
                "relative singular values = "
                + np.array2string(
                    diagnostic["relative_singular_values"],
                    precision=8,
                )
                + "\n"
            )
            handle.write(f"effective ranks = {diagnostic['effective_ranks']}\n")
            handle.write("absolute Gram matrix =\n")
            handle.write(
                np.array2string(
                    np.abs(diagnostic["gram"]),
                    precision=6,
                    suppress_small=True,
                )
            )
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pathway-basis conditioning by response stage."
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
        default=Path("stage_diagnostics"),
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

    stages = ("after_J2", "after_U2", "after_J3", "after_G3")
    sector_sets = [("RP",), ("NRP",), ("RP", "NRP")]

    for sectors in sector_sets:
        sector_label = "_".join(sectors)

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
        stage_results: dict[str, dict] = {}

        print("\n" + "=" * 80)
        print("SECTORS:", sectors)

        for stage in stages:
            matrix = stage_basis_matrix(
                omegas,
                canonical,
                stage,
                tau2=args.tau2,
                w0=args.w0,
                gamma_phi=args.gamma_phi,
                gamma1=args.gamma1,
                eta=args.eta,
                initial=args.initial,
            )

            diagnostic = diagnose_matrix(matrix)
            stage_results[stage] = diagnostic

            print(
                f"{stage:10s}  "
                f"cond={diagnostic['condition_number']:.6e}  "
                f"rank@1e-12="
                f"{diagnostic['effective_ranks'][1e-12]}"
            )

            plot_stage_gram(
                diagnostic,
                labels,
                f"{sector_label}: {STAGE_NAMES[stage]}",
                args.output_dir
                / f"{sector_label}_{stage}_gram_abs.png",
            )

        plot_condition_by_stage(
            stage_results,
            f"{sector_label} pathway-basis conditioning",
            args.output_dir / f"{sector_label}_condition_by_stage.png",
        )

        write_stage_summary(
            args.output_dir / f"{sector_label}_stage_summary.txt",
            sector_label,
            labels,
            stage_results,
        )

    print(f"\nStage diagnostics written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
