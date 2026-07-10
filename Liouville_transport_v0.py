# ============================================================
# Liouville-space pathway transport
# ============================================================
#
# Constructs the pathway states BEFORE optical detection:
#
#   |P_a(lambda; omega)> =
#       G3 J3 U2 J2 G1 J1 |rho0>
#
# and computes the transport matrix from
#
#   |P_a(lambda)> ~= sum_b T_ab(lambda) |P_b(0)>.
#
# Frequency samples are stacked together, so the projection
# uses the full Liouville-space trajectory over the spectral
# window rather than the detected scalar spectrum.
#
# Recommended:
#   initial="fixed_steady0"
#
# This isolates the deformation of the Liouvillian dynamics
# from changes in the initial steady state.
# ============================================================

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt




# Add near the top of Liouville_transport.py

from phase_matched_response import (
    G_freq,
    get_sector_pathways,
    initial_state,
    interaction_super,
    liouvillian,
    omega1_sign_for_sector,
    pathway_label,
)


# ------------------------------------------------------------
# Undetected Liouville-space pathway state
# ------------------------------------------------------------

def pathway_state_point(
    theta,
    omega1,
    omega3,
    pathway,
    omega1_sign=+1,
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
):
    """
    Return the Liouville-space pathway vector before detection.

    Output shape:
        (4,) for the two-level model.
    """
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

    G1 = G_freq(
        L,
        omega1_sign * omega1,
        eta=eta,
    )

    G3 = G_freq(
        L,
        omega3,
        eta=eta,
    )

    U2 = scipy.linalg.expm(L * tau2)

    return G3 @ J3 @ U2 @ J2 @ G1 @ J1 @ rho0


def pathway_state_spectrum(
    theta,
    omegas,
    pathway,
    sector,
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
):
    """
    Compute the undetected pathway state over a diagonal
    frequency slice omega1 = omega3 = omega.

    Returns
    -------
    states : ndarray, shape (n_omega, 4)
        Liouville-space vector at each frequency.
    """
    sector = sector.upper()
    s1 = omega1_sign_for_sector(sector)

    return np.array([
        pathway_state_point(
            theta=theta,
            omega1=omega,
            omega3=omega,
            pathway=pathway,
            omega1_sign=s1,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )
        for omega in omegas
    ])


# ------------------------------------------------------------
# Identify the canonical pathways at lambda = 0
# ------------------------------------------------------------

def active_liouville_pathways(
    sector,
    omegas,
    cutoff=1e-10,
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
):
    """
    Retain pathways having nonzero Liouville-space norm at theta=0.

    This identifies the canonical surviving diagrams rather than
    all formally phase-matched side/field combinations.
    """
    sector = sector.upper()
    active = []

    for pathway in get_sector_pathways(sector):
        states = pathway_state_spectrum(
            theta=0.0,
            omegas=omegas,
            pathway=pathway,
            sector=sector,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )

        norm = np.sqrt(
            np.trapezoid(
                np.sum(np.abs(states)**2, axis=1),
                omegas,
            )
        )

        if norm > cutoff:
            active.append(pathway)

    return active


def canonical_pathway_list(
    omegas,
    sectors=("RP", "NRP"),
    **kwargs,
):
    """
    Assemble one combined canonical pathway list.

    Returns a list of dictionaries:
        {
            "sector": "RP",
            "pathway": ...,
            "label": "RP: L-L+R+"
        }
    """
    canonical = []

    for sector in sectors:
        paths = active_liouville_pathways(
            sector=sector,
            omegas=omegas,
            **kwargs,
        )

        for pathway in paths:
            canonical.append({
                "sector": sector,
                "pathway": pathway,
                "label": f"{sector}: {pathway_label(pathway)}",
            })

    return canonical


# ------------------------------------------------------------
# Build the pathway-state basis matrix
# ------------------------------------------------------------

def liouville_pathway_basis_matrix(
    theta,
    omegas,
    canonical_pathways,
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
    quadrature_weight=True,
):
    """
    Construct a matrix whose columns are complete pathway states.

    Each pathway state has dimensions:

        n_omega x n_Liouville

    and is flattened into one column of length

        n_omega * n_Liouville.

    If quadrature_weight=True, sqrt(domega) weights are included,
    making the Euclidean projection equivalent to an approximate
    frequency integral.
    """
    columns = []

    if quadrature_weight:
        domega = np.gradient(omegas)
        sqrt_weight = np.sqrt(domega)[:, None]
    else:
        sqrt_weight = 1.0

    for item in canonical_pathways:
        states = pathway_state_spectrum(
            theta=theta,
            omegas=omegas,
            pathway=item["pathway"],
            sector=item["sector"],
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            initial=initial,
        )

        weighted_states = sqrt_weight * states
        columns.append(weighted_states.reshape(-1))

    return np.column_stack(columns)


# ------------------------------------------------------------
# Liouville-space transport matrix
# ------------------------------------------------------------

def liouville_transport_matrix(
    theta,
    omegas,
    canonical_pathways,
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
    rcond=1e-12,
):
    """
    Compute T_ab(theta) from the Liouville-space basis projection.

    Let

        P0[:, b]      = |P_b(0)>
        Plambda[:, a] = |P_a(theta)>.

    Solve

        Plambda = P0 @ C,

    where C[b,a] is the coefficient of canonical pathway b in
    transported pathway a. We return

        T[a,b] = C[b,a].

    Thus:

        |P_a(theta)> ~= sum_b T[a,b] |P_b(0)>.
    """
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

    Plam = liouville_pathway_basis_matrix(
        theta=theta,
        omegas=omegas,
        canonical_pathways=canonical_pathways,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        initial=initial,
    )

    # SVD pseudoinverse is stable for a nonorthogonal basis.
    P0_pinv = np.linalg.pinv(P0, rcond=rcond)

    # C[b,a]
    C = P0_pinv @ Plam

    # T[a,b]
    T = C.T

    reconstructed = P0 @ C

    pathway_residuals = np.zeros(Plam.shape[1])

    for a in range(Plam.shape[1]):
        pathway_residuals[a] = (
            np.linalg.norm(Plam[:, a] - reconstructed[:, a])
            / max(np.linalg.norm(Plam[:, a]), 1e-14)
        )

    singular_values = np.linalg.svd(P0, compute_uv=False)

    rank = np.linalg.matrix_rank(
        P0,
        tol=rcond * singular_values[0],
    )

    if singular_values[-1] > 0:
        condition_number = singular_values[0] / singular_values[-1]
    else:
        condition_number = np.inf

    return {
        "T": T,
        "P0": P0,
        "Plambda": Plam,
        "reconstructed": reconstructed,
        "residuals": pathway_residuals,
        "singular_values": singular_values,
        "rank": rank,
        "condition_number": condition_number,
    }


def liouville_transport_sweep(
    lambda_values,
    omegas,
    sectors=("RP", "NRP"),
    tau2=1.0,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    eta=1e-3,
    initial="fixed_steady0",
    cutoff=1e-10,
    rcond=1e-12,
):
    """
    Compute the Liouville-space transport matrix over lambda.
    """
    canonical = canonical_pathway_list(
        omegas=omegas,
        sectors=sectors,
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

    for lam in lambda_values:
        result = liouville_transport_matrix(
            theta=lam,
            omegas=omegas,
            canonical_pathways=canonical,
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

    return {
        "sector": "+".join(sectors),
        "lambda_values": np.asarray(lambda_values),
        "canonical_pathways": canonical,
        "labels": labels,
        "T": np.asarray(T_values),
        "residuals": np.asarray(residual_values),
        "ranks": np.asarray(ranks),
        "condition_numbers": np.asarray(condition_numbers),
    }


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def print_liouville_transport_diagnostics(result):
    print("\nCanonical Liouville pathways")
    print("-" * 70)

    for i, label in enumerate(result["labels"]):
        print(f"{i:2d}: {label}")

    print("\nBasis diagnostics")
    print("-" * 70)

    for lam, rank, cond in zip(
        result["lambda_values"],
        result["ranks"],
        result["condition_numbers"],
    ):
        print(
            f"lambda = {lam:8.5f}   "
            f"rank = {rank:2d}   "
            f"condition number = {cond:12.5e}"
        )

    print("\nIdentity check at lambda = 0")
    print("-" * 70)
    print(np.real_if_close(result["T"][0]))

    identity_error = np.linalg.norm(
        result["T"][0] - np.eye(len(result["labels"]))
    )

    print("||T(0)-I|| =", identity_error)


def plot_liouville_transport_elements(
    result,
    component="abs",
    threshold=0.0,
    figsize=(9, 5),
    savefile=None,
):
    """
    Plot all retained T_ab(lambda) elements.
    """
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
                    "component must be 'abs', 'real', or 'imag'"
                )

            if np.max(np.abs(values)) <= threshold:
                continue

            ax.plot(
                lam,
                values,
                marker="o",
                label=f"{labels[a]} <- {labels[b]}",
            )

    ax.set_xlabel(r"$\lambda \equiv \theta$")
    ax.set_ylabel(ylabel)
    ax.set_title("Liouville-space pathway transport")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    if savefile is not None:
        fig.savefig(
            savefile,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_liouville_transport_heatmap(
    result,
    lambda_index=-1,
    component="abs",
    figsize=(7, 6),
    savefile=None,
):
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
            "component must be 'abs', 'real', or 'imag'"
        )

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="equal",
    )

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel(r"canonical source pathway $b$")
    ax.set_ylabel(r"transported pathway $a$")
    ax.set_title(title)

    fig.tight_layout()

    if savefile is not None:
        fig.savefig(
            savefile,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()