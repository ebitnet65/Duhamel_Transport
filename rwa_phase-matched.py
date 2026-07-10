# ============================================================
# RWA phase-matched Liouville-sector response module
# ============================================================
#
# Purpose
# -------
# Frequency-domain RP/NRP/DQC decomposition for the rotated
# pointer-basis Liouvillian model.
#
# This version correctly separates each optical interaction into:
#
#   1. left/right Liouville action,
#   2. positive/negative dipole component,
#   3. positive/negative field phase.
#
# Phase matching is determined by the field phases, not by the
# left/right Liouville side alone.
#
# Sectors
# -------
# COMM    : original nested-commutator response with full mu
# ALL_RWA : sum over all RWA-resolved Liouville pathways
# RP      : rephasing,       -k1 + k2 + k3
# NRP     : nonrephasing,    +k1 - k2 + k3
# DQC     : double quantum,  +k1 + k2 - k3
#
# RWA convention
# --------------
#   field +  carries +k_j and couples to mu^-
#   field -  carries -k_j and couples to mu^+
#
# If your optical phase convention is opposite, flip FIELD_TO_MU.
# ============================================================

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from itertools import product
from numpy import kron


# ------------------------------------------------------------
# Basic two-level operators
# ------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

sig_x = np.array([[0, 1],
                  [1, 0]], dtype=complex)

sig_z = np.array([[-1, 0],
                  [ 0, 1]], dtype=complex)

sig_p = np.array([[0, 0],
                  [1, 0]], dtype=complex)

sig_m = np.array([[0, 1],
                  [0, 0]], dtype=complex)

mu_plus = sig_p
mu_minus = sig_m
mu = mu_plus + mu_minus


# ------------------------------------------------------------
# Vectorization and superoperators
# ------------------------------------------------------------

def vec(rho):
    """Column-stacking vectorization."""
    return rho.reshape(-1, order="F")


def mat(rho_vec, dim=2):
    """Inverse column-stacking vectorization."""
    return rho_vec.reshape((dim, dim), order="F")


def left_super(O):
    """Left multiplication: rho -> O rho."""
    return kron(I2, O)


def right_super(O):
    """Right multiplication: rho -> rho O."""
    return kron(O.T, I2)


def comm_super(O):
    """Commutator: rho -> [O, rho]."""
    return left_super(O) - right_super(O)


def J_left(O):
    """Left optical action: rho -> O rho."""
    return left_super(O)


def J_right(O):
    """Right optical action with commutator sign: rho -> -rho O."""
    return -right_super(O)


def J_side(side, O):
    side = side.upper()

    if side == "L":
        return J_left(O)

    if side == "R":
        return J_right(O)

    raise ValueError("side must be 'L' or 'R'")


# ------------------------------------------------------------
# RWA optical components
# ------------------------------------------------------------

FIELD_TO_MU = {
    "+": mu_minus,  # E^(+) couples to mu^-
    "-": mu_plus,   # E^(-) couples to mu^+
}

FIELD_PHASE_SIGN = {
    "+": +1,
    "-": -1,
}


def mu_for_field(field):
    field = field.strip()

    if field not in FIELD_TO_MU:
        raise ValueError("field must be '+' or '-'")

    return FIELD_TO_MU[field]


def phase_sign_for_field(field):
    field = field.strip()

    if field not in FIELD_PHASE_SIGN:
        raise ValueError("field must be '+' or '-'")

    return FIELD_PHASE_SIGN[field]


def interaction_super(side, field):
    """
    RWA-resolved optical interaction.

    side  = 'L' or 'R'
    field = '+' or '-'

    The field sign determines the wave-vector phase.
    The dipole component is chosen by the RWA.
    """
    return J_side(side, mu_for_field(field))


# ------------------------------------------------------------
# Liouvillian model
# ------------------------------------------------------------

def liouvillian(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02):
    """
    Rotated pointer-basis two-level Liouvillian.
    """
    H = 0.5 * w0 * sig_z

    L = -1j * (left_super(H) - right_super(H))

    A = np.cos(theta) * sig_z + np.sin(theta) * sig_x
    C = np.sqrt(gamma_phi) * A
    CdC = C.conj().T @ C

    L += (
        kron(C.conj(), C)
        - 0.5 * kron(I2, CdC)
        - 0.5 * kron(CdC.T, I2)
    )

    if gamma1 > 0:
        C1 = np.sqrt(gamma1) * sig_m
        CdC1 = C1.conj().T @ C1

        L += (
            kron(C1.conj(), C1)
            - 0.5 * kron(I2, CdC1)
            - 0.5 * kron(CdC1.T, I2)
        )

    return L


def G_freq(L, omega, eta=1e-3):
    """
    Frequency-domain Liouvillian resolvent.

    Matches the sign convention in the original diagonal sweep.
    """
    n = L.shape[0]
    return -np.linalg.inv(L + (1j * omega - eta) * np.eye(n))


def rho_initial_ground():
    rho0 = np.array([[1, 0],
                     [0, 0]], dtype=complex)
    return vec(rho0)


# ------------------------------------------------------------
# Pathway bookkeeping
# ------------------------------------------------------------

def make_pathway(sides, fields):
    """
    Construct a pathway as a tuple of three interactions.

    sides  = tuple/list of 'L'/'R'
    fields = tuple/list of '+'/'-'

    Returns:
        ( (side1, field1), (side2, field2), (side3, field3) )
    """
    if len(sides) != 3 or len(fields) != 3:
        raise ValueError("sides and fields must both have length 3")

    return tuple((s.upper(), f) for s, f in zip(sides, fields))


def pathway_phase_signature(pathway):
    """
    Return the k-vector phase signature of an RWA-resolved pathway.

    Example:
        fields ('-', '+', '+') -> (-1, +1, +1)
    """
    return tuple(phase_sign_for_field(field) for side, field in pathway)


def pathway_label(pathway):
    """
    Human-readable label.

    Example:
        (('R','-'),('L','+'),('L','+')) -> R-L+L+
    """
    return "".join([f"{side}{field}" for side, field in pathway])


def all_rwa_pathways():
    """
    All RWA-resolved pathways:
        2^3 side choices x 2^3 field choices = 64 pathways.
    """
    pathways = []

    for sides in product(["L", "R"], repeat=3):
        for fields in product(["+", "-"], repeat=3):
            pathways.append(make_pathway(sides, fields))

    return pathways


SECTOR_PHASES = {
    "RP":  (-1, +1, +1),
    "NRP": (+1, -1, +1),
    "DQC": (+1, +1, -1),
}


SECTORS = {
    "COMM": {
        "description": "Original nested-commutator response with full mu",
        "phase": None,
    },

    "ALL_RWA": {
        "description": "Explicit sum over all RWA-resolved pathways",
        "phase": None,
    },

    "RP": {
        "description": "Rephasing sector: -k1 + k2 + k3",
        "phase": SECTOR_PHASES["RP"],
    },

    "NRP": {
        "description": "Nonrephasing sector: +k1 - k2 + k3",
        "phase": SECTOR_PHASES["NRP"],
    },

    "DQC": {
        "description": "Double-quantum sector: +k1 + k2 - k3",
        "phase": SECTOR_PHASES["DQC"],
    },
}


def get_sector_pathways(sector):
    """
    Return RWA-resolved pathways for a sector.

    COMM is not pathway-resolved and should be evaluated separately.
    """
    sector = sector.upper()

    if sector == "COMM":
        raise ValueError("COMM is not pathway-resolved. Use signal_point_diag_comm.")

    pathways = all_rwa_pathways()

    if sector == "ALL_RWA":
        return pathways

    if sector not in SECTOR_PHASES:
        raise ValueError(f"Unknown sector '{sector}'")

    target = SECTOR_PHASES[sector]

    return [
        p for p in pathways
        if pathway_phase_signature(p) == target
    ]


def print_sector_summary(max_print=8):
    print("Available sectors:\n")

    for name, data in SECTORS.items():
        print(f"{name:8s}: {data['description']}")

        if name not in ["COMM"]:
            paths = get_sector_pathways(name)
            print(f"          number of pathways: {len(paths)}")

            for p in paths[:max_print]:
                print(f"          {pathway_label(p):8s}  phase {pathway_phase_signature(p)}")

            if len(paths) > max_print:
                print(f"          ... {len(paths)-max_print} more")

        print("")


# ------------------------------------------------------------
# Frequency-domain response
# ------------------------------------------------------------

def signal_point_diag_comm(theta, omega,
                           tau2=1.0,
                           w0=1.0,
                           gamma_phi=0.1,
                           gamma1=0.02,
                           eta=1e-3):
    """
    Original nested-commutator diagonal response using full mu.
    """
    L = liouvillian(
        w0=w0,
        gamma_phi=gamma_phi,
        theta=theta,
        gamma1=gamma1,
    )

    J = comm_super(mu)
    rho0 = rho_initial_ground()

    G = G_freq(L, omega, eta=eta)
    U2 = scipy.linalg.expm(L * tau2)

    rho_vec = G @ J @ U2 @ J @ G @ J @ rho0
    rho = mat(rho_vec, dim=2)

    return np.trace(mu @ rho)


def signal_point_diag_pathway(theta, omega, pathway,
                              tau2=1.0,
                              w0=1.0,
                              gamma_phi=0.1,
                              gamma1=0.02,
                              eta=1e-3,
                              detection="full"):
    """
    Frequency-domain diagonal response for a single RWA-resolved pathway.

    pathway:
        ((side1, field1), (side2, field2), (side3, field3))

    detection:
        "full"  -> Tr[mu rho]
        "+"     -> Tr[mu_plus rho]
        "-"     -> Tr[mu_minus rho]

    For phase-matched optical heterodyne detection, one may want to
    select a detection component. The default reproduces the earlier
    full-mu observable.
    """
    if len(pathway) != 3:
        raise ValueError("pathway must contain three interactions")

    L = liouvillian(
        w0=w0,
        gamma_phi=gamma_phi,
        theta=theta,
        gamma1=gamma1,
    )

    J1 = interaction_super(*pathway[0])
    J2 = interaction_super(*pathway[1])
    J3 = interaction_super(*pathway[2])

    rho0 = rho_initial_ground()

    G1 = G_freq(L, omega, eta=eta)
    G3 = G_freq(L, omega, eta=eta)
    U2 = scipy.linalg.expm(L * tau2)

    rho_vec = G3 @ J3 @ U2 @ J2 @ G1 @ J1 @ rho0
    rho = mat(rho_vec, dim=2)

    if detection == "full":
        Odet = mu
    elif detection == "+":
        Odet = mu_plus
    elif detection == "-":
        Odet = mu_minus
    else:
        raise ValueError("detection must be 'full', '+', or '-'")

    return np.trace(Odet @ rho)


def signal_point_diag(theta, omega, sector="COMM",
                      tau2=1.0,
                      w0=1.0,
                      gamma_phi=0.1,
                      gamma1=0.02,
                      eta=1e-3,
                      detection="full"):
    """
    Sector-resolved diagonal response.

    sector:
        COMM
        ALL_RWA
        RP
        NRP
        DQC
    """
    sector = sector.upper()

    if sector == "COMM":
        return signal_point_diag_comm(
            theta, omega,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
        )

    total = 0.0 + 0.0j

    for pathway in get_sector_pathways(sector):
        total += signal_point_diag_pathway(
            theta,
            omega,
            pathway,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            detection=detection,
        )

    return total


def signal_sweep(theta_values, omegas, sector="COMM",
                 tau2=1.0,
                 w0=1.0,
                 gamma_phi=0.1,
                 gamma1=0.02,
                 eta=1e-3,
                 detection="full"):
    out = []

    for th in theta_values:
        S_th = np.array([
            signal_point_diag(
                th,
                om,
                sector=sector,
                tau2=tau2,
                w0=w0,
                gamma_phi=gamma_phi,
                gamma1=gamma1,
                eta=eta,
                detection=detection,
            )
            for om in omegas
        ])

        out.append(S_th)

    return np.array(out)


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def check_comm_vs_all_rwa(theta=np.pi / 4,
                          omega=1.0,
                          tau2=1.0,
                          w0=1.0,
                          gamma_phi=0.1,
                          gamma1=0.02,
                          eta=1e-3,
                          detection="full",
                          verbose=True):
    """
    Compare original COMM response against ALL_RWA.

    Important:
        COMM uses full mu at every interaction.
        ALL_RWA resolves the three field interactions into RWA pieces
        and uses the chosen detection component.

    Agreement is expected only if the RWA-resolved sum plus detection
    convention reproduces the original full-mu response.
    If not, compare ALL_RWA with an explicitly RWA-constructed total.
    """
    S_comm = signal_point_diag(
        theta, omega,
        sector="COMM",
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        detection=detection,
    )

    S_all = signal_point_diag(
        theta, omega,
        sector="ALL_RWA",
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        detection=detection,
    )

    mismatch = np.abs(S_comm - S_all) / max(np.abs(S_comm), 1e-14)

    if verbose:
        print("COMM response:    ", S_comm)
        print("ALL_RWA response: ", S_all)
        print("relative mismatch:", mismatch)

    return mismatch, S_comm, S_all


def deformation_metric(signals, omegas):
    ref = signals[0]

    return np.array([
        np.trapezoid(np.abs(S - ref)**2, omegas)
        for S in signals
    ])


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def plot_sector_sweep(omegas, signals, theta_labels,
                      sector="COMM",
                      component="real",
                      filename=None):
    component = component.lower()

    plt.figure(figsize=(8, 5))

    for S, lab in zip(signals, theta_labels):
        if component == "real":
            y = np.real(S)
            ylabel = rf"$\mathrm{{Re}}\,S^{{(3)}}(\omega,\omega)$"
        elif component == "imag":
            y = np.imag(S)
            ylabel = rf"$\mathrm{{Im}}\,S^{{(3)}}(\omega,\omega)$"
        elif component == "abs":
            y = np.abs(S)
            ylabel = rf"$|S^{{(3)}}(\omega,\omega)|$"
        else:
            raise ValueError("component must be 'real', 'imag', or 'abs'")

        plt.plot(omegas, y, label=lab)

    plt.xlabel(r"$\omega = \omega_1 = \omega_3$")
    plt.ylabel(ylabel)
    plt.legend(title=r"$\theta$")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)

    plt.show()


def plot_deformation(theta_values, deformation_by_sector, filename=None):
    plt.figure(figsize=(7, 4.5))

    for sector, deform in deformation_by_sector.items():
        plt.plot(theta_values, deform, marker="o", label=sector)

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\int d\omega\, |S(\theta)-S(0)|^2$")
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)

    plt.show()


# ------------------------------------------------------------
# Example Colab driver
# ------------------------------------------------------------

def run_phase_matched_diagnostic():
    w0 = 1.0
    omega0 = 1.0
    gamma_phi = 0.1
    gamma1 = 0.02
    tau2 = 1.0
    eta = 1e-3

    delta = 0.75
    omegas = np.linspace(omega0 - delta, omega0 + delta, 1001)

    theta_values = np.array([
        0,
        np.pi / 8,
        np.pi / 4,
        3 * np.pi / 8,
        np.pi / 2,
    ])

    theta_labels = [
        "0",
        r"$\pi/8$",
        r"$\pi/4$",
        r"$3\pi/8$",
        r"$\pi/2$",
    ]

    print_sector_summary()

    print("Checking COMM vs ALL_RWA...")
    check_comm_vs_all_rwa(
        theta=np.pi / 4,
        omega=omega0,
        tau2=tau2,
        w0=w0,
        gamma_phi=gamma_phi,
        gamma1=gamma1,
        eta=eta,
        detection="full",
        verbose=True,
    )

    sectors_to_compute = ["COMM", "ALL_RWA", "RP", "NRP"]

    signals_by_sector = {}

    for sector in sectors_to_compute:
        print(f"Computing sector: {sector}")

        signals_by_sector[sector] = signal_sweep(
            theta_values,
            omegas,
            sector=sector,
            tau2=tau2,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            eta=eta,
            detection="full",
        )

        plot_sector_sweep(
            omegas,
            signals_by_sector[sector],
            theta_labels,
            sector=sector,
            component="real",
            filename=f"diag_theta_{sector}_real.png",
        )

        plot_sector_sweep(
            omegas,
            signals_by_sector[sector],
            theta_labels,
            sector=sector,
            component="imag",
            filename=f"diag_theta_{sector}_imag.png",
        )

    deformation_by_sector = {
        sector: deformation_metric(signals, omegas)
        for sector, signals in signals_by_sector.items()
    }

    plot_deformation(
        theta_values,
        deformation_by_sector,
        filename="phase_matched_sector_deformation_RWA.png",
    )

    return {
        "omegas": omegas,
        "theta_values": theta_values,
        "theta_labels": theta_labels,
        "signals_by_sector": signals_by_sector,
        "deformation_by_sector": deformation_by_sector,
    }