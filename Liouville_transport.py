"""
Liouville_transport.py

Liouville-space pathway transport for the rotated-pointer-basis
two-level model.

The transport basis is defined at a selectable stage of the
third-order response sequence:

    after_J2 : J2 G1 J1 rho0
    after_U2 : U2 J2 G1 J1 rho0
    after_J3 : J3 U2 J2 G1 J1 rho0
    after_G3 : G3 J3 U2 J2 G1 J1 rho0

For the two-level model, the recommended transport stage is
``after_U2`` because the canonical GSB/SE pathway manifold remains
rank two there, whereas the third interaction collapses the pathways
onto a common emitting-coherence direction.

The finite-lambda pathway states are projected onto the independent
lambda=0 pathway basis:

    |P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>.

Frequency samples are stacked and quadrature weighted so that the
Euclidean projection approximates the frequency-integrated
Liouville-space inner product.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from phase_matched_response import (
    G_freq,
    get_sector_pathways,
    initial_state,
    interaction_super,
    liouvillian,
    omega1_sign_for_sector,
    pathway_label,
)

VALID_STAGES = ("after_J2", "after_U2", "after_J3", "after_G3")

PHYSICAL_PATHWAY_LABELS = {
    ("RP", "L-L+L+"): "RP-GSB",
    ("RP", "L-R+L+"): "RP-SE",
    ("NRP", "L+L-L+"): "NRP-GSB",
    ("NRP", "L+R-L+"): "NRP-SE",
}


def physical_pathway_label(sector, pathway):
    raw = pathway_label(pathway)
    return PHYSICAL_PATHWAY_LABELS.get(
        (sector.upper(), raw),
        f"{sector.upper()}-{raw}",
    )


def _validate_stage(stage: str) -> str:
    stage = stage.strip()
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}. Expected one of {VALID_STAGES}."
        )
    return stage


def pathway_state_point(
    theta: float,
    omega1: float,
    omega3: float,
    pathway: Sequence[tuple[str, str]],
    sector: str,
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
) -> np.ndarray:
    """Return one Liouville-space pathway state at a selected stage."""
    stage = _validate_stage(stage)
    sector = sector.upper()

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
    rho0 = initial_state(L, mode=initial, L_ref=L_ref)

    J1 = interaction_super(*pathway[0])
    J2 = interaction_super(*pathway[1])
    J3 = interaction_super(*pathway[2])

    s1 = omega1_sign_for_sector(sector)
    G1 = G_freq(L, s1 * omega1, eta=eta)
    G3 = G_freq(L, omega3, eta=eta)
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

    return G3 @ state


def pathway_state_spectrum(
    theta: float,
    omegas: np.ndarray,
    pathway: Sequence[tuple[str, str]],
    sector: str,
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
) -> np.ndarray:
    """Compute a pathway Liouville state over omega1=omega3=omega."""
    omegas = np.asarray(omegas, dtype=float)
    return np.asarray([
        pathway_state_point(
            theta=theta,
            omega1=omega,
            omega3=omega,
            pathway=pathway,
            sector=sector,
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


def active_liouville_pathways(
    sector: str,
    omegas: np.ndarray,
    *,
    stage: str = "after_U2",
    cutoff: float = 1e-10,
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
) -> list:
    """Retain pathways with nonzero lambda=0 norm at the selected stage."""
    stage = _validate_stage(stage)
    sector = sector.upper()
    active = []

    for pathway in get_sector_pathways(sector):
        states = pathway_state_spectrum(
            theta=0.0,
            omegas=omegas,
            pathway=pathway,
            sector=sector,
            stage=stage,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        norm = np.sqrt(
            np.trapezoid(np.sum(np.abs(states) ** 2, axis=1), omegas)
        )
        if norm > cutoff:
            active.append(pathway)

    return active


def canonical_pathway_list(
    omegas: np.ndarray,
    sectors: Iterable[str] = ("RP", "NRP"),
    *,
    stage: str = "after_U2",
    cutoff: float = 1e-10,
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
) -> list[dict[str, Any]]:
    """Assemble the nonzero canonical pathway candidates."""
    canonical: list[dict[str, Any]] = []

    for sector in sectors:
        sector = sector.upper()

        paths = active_liouville_pathways(
            sector=sector,
            omegas=omegas,
            stage=stage,
            cutoff=cutoff,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )

        for pathway in paths:
            canonical.append({
                "sector": sector,
                "pathway": pathway,
                "label": physical_pathway_label(
                    sector,
                    pathway,
                ),
            })

    return canonical

for sector in ("RP", "NRP"):
    print(sector)

    for pathway in get_sector_pathways(sector):
        print(pathway_label(pathway))


def liouville_pathway_basis_matrix(
    theta: float,
    omegas: np.ndarray,
    canonical_pathways: Sequence[dict[str, Any]],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
    quadrature_weight: bool = True,
) -> np.ndarray:
    """Build the pathway-state matrix with pathways as columns."""
    stage = _validate_stage(stage)
    omegas = np.asarray(omegas, dtype=float)

    if len(canonical_pathways) == 0:
        raise ValueError("canonical_pathways is empty.")

    sqrt_weight = (
        np.sqrt(np.gradient(omegas))[:, None]
        if quadrature_weight
        else np.ones((len(omegas), 1))
    )

    columns = []
    for item in canonical_pathways:
        states = pathway_state_spectrum(
            theta=theta,
            omegas=omegas,
            pathway=item["pathway"],
            sector=item["sector"],
            stage=stage,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        columns.append((sqrt_weight * states).reshape(-1))

    return np.column_stack(columns)


def select_independent_columns(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    rtol: float = 1e-10,
) -> tuple[list[int], list[str], np.ndarray]:
    """Greedily select linearly independent columns."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    if matrix.shape[1] != len(labels):
        raise ValueError("Number of labels must match matrix columns.")

    selected_indices: list[int] = []
    selected_labels: list[str] = []
    candidate_residuals = np.zeros(matrix.shape[1], dtype=float)
    basis = np.empty((matrix.shape[0], 0), dtype=complex)

    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        column_norm = np.linalg.norm(column)

        if column_norm < 1e-15:
            continue

        if basis.shape[1] == 0:
            selected_indices.append(index)
            selected_labels.append(labels[index])
            basis = column[:, None]
            candidate_residuals[index] = 1.0
            continue

        coefficients, *_ = np.linalg.lstsq(basis, column, rcond=None)
        residual = column - basis @ coefficients
        relative_residual = np.linalg.norm(residual) / column_norm
        candidate_residuals[index] = relative_residual

        if relative_residual > rtol:
            selected_indices.append(index)
            selected_labels.append(labels[index])
            basis = np.column_stack((basis, column))

    return selected_indices, selected_labels, candidate_residuals


def independent_canonical_pathways(
    omegas: np.ndarray,
    sectors: Iterable[str] = ("RP", "NRP"),
    *,
    stage: str = "after_U2",
    independence_mode: str = "by_sector",
    rtol: float = 1e-10,
    cutoff: float = 1e-10,
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Construct a rank-reduced canonical pathway list."""
    stage = _validate_stage(stage)
    independence_mode = independence_mode.lower()

    if independence_mode not in ("by_sector", "combined"):
        raise ValueError(
            "independence_mode must be 'by_sector' or 'combined'."
        )

    sectors = tuple(sector.upper() for sector in sectors)

    def reduce_group(group: list[dict[str, Any]]):
        labels = [item["label"] for item in group]
        matrix = liouville_pathway_basis_matrix(
            theta=0.0,
            omegas=omegas,
            canonical_pathways=group,
            stage=stage,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        indices, selected_labels, residuals = select_independent_columns(
            matrix, labels, rtol=rtol
        )
        return [group[index] for index in indices], selected_labels, residuals

    if independence_mode == "combined":
        candidates = canonical_pathway_list(
            omegas=omegas,
            sectors=sectors,
            stage=stage,
            cutoff=cutoff,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        selected, selected_labels, _ = reduce_group(candidates)
        if verbose:
            print(f"\nIndependent pathways at stage={stage}")
            print("-" * 72)
            for label in selected_labels:
                print(label)
        return selected

    selected_all: list[dict[str, Any]] = []
    for sector in sectors:
        candidates = canonical_pathway_list(
            omegas=omegas,
            sectors=(sector,),
            stage=stage,
            cutoff=cutoff,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        selected, selected_labels, _ = reduce_group(candidates)
        selected_all.extend(selected)

        if verbose:
            print(f"\nIndependent {sector} pathways at stage={stage}")
            print("-" * 72)
            for label in selected_labels:
                print(label)

    return selected_all


def liouville_transport_matrix(
    theta: float,
    omegas: np.ndarray,
    canonical_pathways: Sequence[dict[str, Any]],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
    rcond: float = 1e-12,
) -> dict[str, Any]:
    """Compute T_ab(theta) in the selected pathway manifold."""
    stage = _validate_stage(stage)

    P0 = liouville_pathway_basis_matrix(
        theta=0.0,
        omegas=omegas,
        canonical_pathways=canonical_pathways,
        stage=stage,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        initial=initial,
    )
    Plam = liouville_pathway_basis_matrix(
        theta=theta,
        omegas=omegas,
        canonical_pathways=canonical_pathways,
        stage=stage,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        initial=initial,
    )

    singular_values = np.linalg.svd(P0, compute_uv=False)
    if singular_values.size == 0:
        raise RuntimeError("Reference basis has no singular values.")

    relative_singular_values = singular_values / singular_values[0]
    rank = int(np.sum(relative_singular_values > rcond))

    if rank < P0.shape[1]:
        raise np.linalg.LinAlgError(
            "Reference pathway basis is rank deficient at "
            f"stage={stage}: rank={rank}, columns={P0.shape[1]}. "
            "Use independent_canonical_pathways() or increase rcond."
        )

    P0_pinv = np.linalg.pinv(P0, rcond=rcond)
    C = P0_pinv @ Plam
    T = C.T
    reconstructed = P0 @ C

    pathway_residuals = np.array([
        np.linalg.norm(Plam[:, a] - reconstructed[:, a])
        / max(np.linalg.norm(Plam[:, a]), 1e-14)
        for a in range(Plam.shape[1])
    ])

    condition_number = (
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 0
        else np.inf
    )

    return {
        "T": T,
        "P0": P0,
        "Plambda": Plam,
        "reconstructed": reconstructed,
        "residuals": pathway_residuals,
        "singular_values": singular_values,
        "relative_singular_values": relative_singular_values,
        "rank": rank,
        "condition_number": condition_number,
        "stage": stage,
    }


def liouville_transport_sweep(
    lambda_values: np.ndarray,
    omegas: np.ndarray,
    sectors: Iterable[str] = ("RP", "NRP"),
    *,
    stage: str = "after_U2",
    independence_mode: str = "by_sector",
    select_independent: bool = True,
    independence_rtol: float = 1e-10,
    cutoff: float = 1e-10,
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
    rcond: float = 1e-12,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute the Liouville-space transport matrix over lambda."""
    stage = _validate_stage(stage)
    sectors = tuple(sector.upper() for sector in sectors)

    if select_independent:
        canonical = independent_canonical_pathways(
            omegas=omegas,
            sectors=sectors,
            stage=stage,
            independence_mode=independence_mode,
            rtol=independence_rtol,
            cutoff=cutoff,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
            verbose=verbose,
        )
    else:
        canonical = canonical_pathway_list(
            omegas=omegas,
            sectors=sectors,
            stage=stage,
            cutoff=cutoff,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )

    labels = [item["label"] for item in canonical]

    T_values = []
    residual_values = []
    ranks = []
    condition_numbers = []
    singular_values = []

    for lam in np.asarray(lambda_values, dtype=float):
        result = liouville_transport_matrix(
            theta=lam,
            omegas=omegas,
            canonical_pathways=canonical,
            stage=stage,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
            rcond=rcond,
        )

        T_values.append(result["T"])
        residual_values.append(result["residuals"])
        ranks.append(result["rank"])
        condition_numbers.append(result["condition_number"])
        singular_values.append(result["singular_values"])

    return {
        "sector": "+".join(sectors),
        "stage": stage,
        "lambda_values": np.asarray(lambda_values, dtype=float),
        "canonical_pathways": canonical,
        "labels": labels,
        "T": np.asarray(T_values),
        "residuals": np.asarray(residual_values),
        "ranks": np.asarray(ranks),
        "condition_numbers": np.asarray(condition_numbers),
        "singular_values": np.asarray(singular_values),
    }


def diagnose_reference_basis(
    omegas: np.ndarray,
    canonical_pathways: Sequence[dict[str, Any]],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    w0: float = 1.0,
    gamma_phi: float = 0.1,
    gamma1: float = 0.02,
    eta: float = 1e-3,
    initial: str = "fixed_steady0",
) -> dict[str, Any]:
    """Diagnose the normalized reference basis at lambda=0."""
    P0 = liouville_pathway_basis_matrix(
        theta=0.0,
        omegas=omegas,
        canonical_pathways=canonical_pathways,
        stage=stage,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        initial=initial,
    )

    norms = np.linalg.norm(P0, axis=0)
    Pn = P0 / np.maximum(norms, 1e-15)[None, :]
    _, singular_values, Vh = np.linalg.svd(Pn, full_matrices=False)
    relative_singular_values = singular_values / singular_values[0]
    gram = Pn.conj().T @ Pn

    return {
        "P0": P0,
        "Pn": Pn,
        "norms": norms,
        "gram": gram,
        "singular_values": singular_values,
        "relative_singular_values": relative_singular_values,
        "right_singular_vectors": Vh,
        "condition_number": (
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0
            else np.inf
        ),
        "stage": stage,
    }


def print_reference_basis_diagnostics(
    diagnostic: dict[str, Any], labels: Sequence[str]
) -> None:
    print(f"\nReference-basis diagnostics: {diagnostic['stage']}")
    print("-" * 72)
    for label, norm in zip(labels, diagnostic["norms"]):
        print(f"{label:28s} ||P||={norm:14.6e}")
    print("\nRelative singular values:")
    print(diagnostic["relative_singular_values"])
    print("\nCondition number:")
    print(diagnostic["condition_number"])
    print("\nAbsolute Gram matrix:")
    print(np.array2string(
        np.abs(diagnostic["gram"]), precision=6, suppress_small=True
    ))


def print_liouville_transport_diagnostics(
    result: dict[str, Any],
) -> None:
    print("\nCanonical Liouville pathways")
    print("-" * 72)
    for index, label in enumerate(result["labels"]):
        print(f"{index:2d}: {label}")

    print(f"\nTransport stage: {result['stage']}")
    print("-" * 72)
    for lam, rank, cond, residuals in zip(
        result["lambda_values"],
        result["ranks"],
        result["condition_numbers"],
        result["residuals"],
    ):
        print(
            f"lambda={lam:9.6f}  rank={rank:2d}  "
            f"cond={cond:12.5e}  "
            f"max residual={np.max(residuals):12.5e}"
        )

    T0 = result["T"][0]
    identity_error = np.linalg.norm(
        T0 - np.eye(T0.shape[0], dtype=complex)
    )

    print("\nIdentity check")
    print("-" * 72)
    print(np.real_if_close(T0))
    print(f"||T(0)-I|| = {identity_error:.12e}")


def plot_liouville_transport_elements(
    result: dict[str, Any],
    *,
    component: str = "abs",
    threshold: float = 0.0,
    figsize: tuple[float, float] = (9, 5),
    savefile: str | None = None,
    show: bool = True,
) -> None:
    lam = result["lambda_values"]
    T = result["T"]
    labels = result["labels"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    for a in range(n):
        for b in range(n):
            if component == "abs":
                values = np.abs(T[:, a, b])
                ylabel = r"$|T_{ab}(\lambda)|$"
            elif component == "real":
                values = np.real(T[:, a, b])
                ylabel = r"$\mathrm{Re}\,T_{ab}(\lambda)$"
            elif component == "imag":
                values = np.imag(T[:, a, b])
                ylabel = r"$\mathrm{Im}\,T_{ab}(\lambda)$"
            else:
                raise ValueError(
                    "component must be 'abs', 'real', or 'imag'."
                )

            if np.max(np.abs(values)) <= threshold:
                continue

            ax.plot(
                lam,
                values,
                marker="o",
                markersize=3,
                label=f"{labels[a]} <- {labels[b]}",
            )

    ax.set_xlabel(r"$\lambda \equiv \theta$")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Liouville-space pathway transport ({result['stage']})"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_liouville_transport_heatmap(
    result: dict[str, Any],
    *,
    lambda_index: int = -1,
    component: str = "abs",
    figsize: tuple[float, float] = (7, 6),
    savefile: str | None = None,
    show: bool = True,
) -> None:
    T = result["T"][lambda_index]
    labels = result["labels"]
    lam = result["lambda_values"][lambda_index]

    if component == "abs":
        matrix = np.abs(T)
        title = rf"$|T_{{ab}}|$, $\lambda={lam:.3f}$"
    elif component == "real":
        matrix = np.real(T)
        title = rf"$\mathrm{{Re}}\,T_{{ab}}$, $\lambda={lam:.3f}$"
    elif component == "imag":
        matrix = np.imag(T)
        title = rf"$\mathrm{{Im}}\,T_{{ab}}$, $\lambda={lam:.3f}$"
    else:
        raise ValueError(
            "component must be 'abs', 'real', or 'imag'."
        )

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, origin="lower", aspect="equal")
    fig.colorbar(image, ax=ax)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"canonical source pathway $b$")
    ax.set_ylabel(r"transported pathway $a$")
    ax.set_title(f"{title}, stage={result['stage']}")

    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = [
    "VALID_STAGES",
    "pathway_state_point",
    "pathway_state_spectrum",
    "active_liouville_pathways",
    "canonical_pathway_list",
    "liouville_pathway_basis_matrix",
    "select_independent_columns",
    "independent_canonical_pathways",
    "liouville_transport_matrix",
    "liouville_transport_sweep",
    "diagnose_reference_basis",
    "print_reference_basis_diagnostics",
    "print_liouville_transport_diagnostics",
    "plot_liouville_transport_elements",
    "plot_liouville_transport_heatmap",
]
