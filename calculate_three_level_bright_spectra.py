"""Full absorptive 2D spectra for a three-level bright-state model.

The model retains |G>, |B1>, and |B2> from the one-exciton manifold of the
SAKE dimer parameterization, but omits the double-exciton state. A one-way
Lindblad channel B2 -> B1 represents internal conversion between the bright
states. The script is a finite-dimensional, finite-difference-branch
calculation; it does not import the SAKE autodiff implementation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg

from phase_matched_response import get_sector_pathways, omega1_sign_for_sector


OMEGA_SITE_1 = 1.0
OMEGA_SITE_2 = 1.18
J_SITE = 0.045
MU_SITE_1 = 1.0
MU_SITE_2 = 1.0
GAMMA_B1 = 0.03
GAMMA_B2 = 0.004
GAMMA_PHI_B1 = 0.02
GAMMA_PHI_B2 = 0.012
ETA = 2e-3
KAPPAS = np.array([0.0, 0.02, 0.06])
T2S = np.array([0.2, 1.0, 5.0, 10.0])


def make_model(kappa: float):
    """Return the bright-basis Hamiltonian, Liouvillian, and dipole pieces."""
    h_site = np.array([[OMEGA_SITE_1, J_SITE],
                       [J_SITE, OMEGA_SITE_2]], dtype=float)
    energies, vectors = np.linalg.eigh(h_site)
    # B1 and B2 are ordered by increasing energy. Transition moments inherit
    # the parallel equal site-dipole convention from the SAKE dimer model.
    d_bright = np.array([MU_SITE_1, MU_SITE_2], dtype=float) @ vectors
    d_bright = np.real_if_close(d_bright).astype(float)

    dim = 3
    eye = np.eye(dim, dtype=complex)
    H = np.diag([0.0, energies[0], energies[1]]).astype(complex)
    mu_plus = np.zeros((dim, dim), dtype=complex)
    mu_plus[1, 0] = d_bright[0]
    mu_plus[2, 0] = d_bright[1]
    mu_minus = mu_plus.conj().T
    mu = mu_plus + mu_minus

    def dissipator(C):
        cd_c = C.conj().T @ C
        return (np.kron(C.conj(), C)
                - 0.5 * np.kron(eye, cd_c)
                - 0.5 * np.kron(cd_c.T, eye))

    L = -1j * (np.kron(eye, H) - np.kron(H.T, eye))
    for idx, rate in ((1, GAMMA_B1), (2, GAMMA_B2)):
        if rate > 0:
            C = np.zeros((dim, dim), dtype=complex)
            C[0, idx] = np.sqrt(rate)
            L += dissipator(C)
    for idx, rate in ((1, GAMMA_PHI_B1), (2, GAMMA_PHI_B2)):
        if rate > 0:
            C = np.zeros((dim, dim), dtype=complex)
            C[idx, idx] = np.sqrt(rate)
            L += dissipator(C)
    if kappa > 0:
        C = np.zeros((dim, dim), dtype=complex)
        C[1, 2] = np.sqrt(kappa)  # one-way internal conversion B2 -> B1
        L += dissipator(C)

    return dict(H=H, L=L, mu=mu, mu_plus=mu_plus, mu_minus=mu_minus,
                energies=energies, eigenvectors=vectors,
                transition_dipoles=d_bright)


def vec(rho):
    return np.asarray(rho).reshape(-1, order="F")


def interaction_super(side: str, field: str, model):
    """Use the field and left/right conventions of phase_matched_response."""
    O = model["mu_minus"] if field == "+" else model["mu_plus"]
    dim = model["H"].shape[0]
    if side.upper() == "L":
        return np.kron(np.eye(dim), O)
    if side.upper() == "R":
        return -np.kron(O.T, np.eye(dim))
    raise ValueError("side must be L or R")


def g_freq(L, omega):
    n = L.shape[0]
    return -np.linalg.inv(L + (1j * omega - ETA) * np.eye(n))


def sector_map(kappa, t2, omegas, sector):
    model = make_model(kappa)
    L = model["L"]
    rho0 = vec(np.diag([1.0, 0.0, 0.0]).astype(complex))
    U2 = scipy.linalg.expm(L * t2)
    signed = -np.asarray(omegas)
    omega1_sign = omega1_sign_for_sector(sector)
    G1 = np.asarray([g_freq(L, omega1_sign * w) for w in signed])
    G3 = np.asarray([g_freq(L, w) for w in signed])
    dim = model["H"].shape[0]
    detect = np.array([
        np.trace(model["mu"] @ np.reshape(np.eye(dim * dim)[:, j],
                                           (dim, dim), order="F"))
        for j in range(dim * dim)
    ])
    S = np.zeros((len(omegas), len(omegas)), dtype=complex)
    for path in get_sector_pathways(sector):
        J1, J2, J3 = [interaction_super(*entry, model) for entry in path]
        first = np.einsum("wij,j->wi", G1, J1 @ rho0)
        second = np.einsum("ij,wj->wi", J2, first)
        waiting = np.einsum("ij,wj->wi", U2, second)
        third = np.einsum("ij,wj->wi", J3, waiting)
        for i, state in enumerate(third):
            detected = np.einsum("wij,j->wi", G3, state)
            # Store matrices as [omega3, omega1] to match image axes.
            S[:, i] += detect @ detected.T
    return S, model


def point_sector(kappa, t2, omega1, omega3, sector):
    """Independent pointwise evaluation for map-convention diagnostics."""
    model = make_model(kappa)
    L = model["L"]
    rho0 = vec(np.diag([1.0, 0.0, 0.0]).astype(complex))
    U2 = scipy.linalg.expm(L * t2)
    w1_signed, w3_signed = -omega1, -omega3
    G1 = g_freq(L, omega1_sign_for_sector(sector) * w1_signed)
    G3 = g_freq(L, w3_signed)
    total = 0j
    for path in get_sector_pathways(sector):
        J1, J2, J3 = [interaction_super(*entry, model) for entry in path]
        rho = G3 @ J3 @ U2 @ J2 @ G1 @ J1 @ rho0
        total += np.trace(model["mu"] @ np.reshape(rho, (3, 3), order="F"))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-omega", type=int, default=81)
    parser.add_argument("--kappas", type=float, nargs="+", default=KAPPAS.tolist())
    parser.add_argument("--t2", type=float, nargs="+", default=T2S.tolist())
    parser.add_argument("--omega-min", type=float, default=0.78)
    parser.add_argument("--omega-max", type=float, default=1.38)
    parser.add_argument("--out", default="three_level_bright_spectra")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    omegas = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    kappas = np.asarray(args.kappas, dtype=float)
    t2s = np.asarray(args.t2, dtype=float)
    rp = np.empty((len(kappas), len(t2s), len(omegas), len(omegas)), complex)
    nrp = np.empty_like(rp)
    absorptive = np.empty(rp.shape, float)
    t_start = time.time()
    model = make_model(float(kappas[0]))
    for ik, kappa in enumerate(kappas):
        for it, t2 in enumerate(t2s):
            rp[ik, it], _ = sector_map(float(kappa), float(t2), omegas, "RP")
            nrp[ik, it], _ = sector_map(float(kappa), float(t2), omegas, "NRP")
            absorptive[ik, it] = -(rp[ik, it] + nrp[ik, it]).real

    diagnostics = []
    for sector, maps in (("RP", rp), ("NRP", nrp)):
        for ik, kappa in enumerate(kappas):
            for it, t2 in enumerate(t2s):
                i1 = len(omegas) // 3
                i3 = 2 * len(omegas) // 3
                direct = point_sector(float(kappa), float(t2),
                                      float(omegas[i1]), float(omegas[i3]), sector)
                diagnostics.append(dict(
                    sector=sector, kappa=float(kappa), t2=float(t2),
                    map_point=[float(maps[ik, it, i3, i1].real),
                               float(maps[ik, it, i3, i1].imag)],
                    direct_point=[float(direct.real), float(direct.imag)],
                    absolute_difference=float(abs(maps[ik, it, i3, i1] - direct)),
                ))

    np.savez_compressed(
        out / "three_level_response_maps.npz",
        omegas=omegas, kappas=kappas, t2s=t2s,
        rephasing=rp, nonrephasing=nrp, absorptive=absorptive,
    )
    vals, vecs = np.linalg.eigh(model["H"][1:, 1:])
    summary = dict(
        model="three-level bright-state truncation of the SAKE exciton dimer one-exciton manifold",
        basis=["G", "B1", "B2"],
        site_parameters=dict(omega1=OMEGA_SITE_1, omega2=OMEGA_SITE_2,
                             J12=J_SITE, mu1=MU_SITE_1, mu2=MU_SITE_2),
        bright_energies=vals.tolist(),
        bright_eigenvectors_site_basis=vecs.real.tolist(),
        transition_dipoles=make_model(0.0)["transition_dipoles"].tolist(),
        ground_relaxation_rates=[GAMMA_B1, GAMMA_B2],
        pure_dephasing_rates=[GAMMA_PHI_B1, GAMMA_PHI_B2],
        interbright_channel="sqrt(kappa)|B1><B2| (one-way B2 to B1)",
        kappas=kappas.tolist(), waiting_times=t2s.tolist(),
        frequency_range=[float(omegas[0]), float(omegas[-1])],
        eta=ETA,
        spectral_convention="positive display axes use signed Fourier arguments -omega1,-omega3; absorptive=-Re(RP+NRP)",
        omitted_state="double-exciton state, so no excited-state absorption is included",
        pointwise_checks=diagnostics,
        maximum_pointwise_difference=max(d["absolute_difference"] for d in diagnostics),
        elapsed_s=time.time() - t_start,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [omegas[0], omegas[-1], omegas[0], omegas[-1]]
    # Plot waiting-time maps for the largest transfer rate with common scale.
    ik = len(kappas) - 1
    vmin = float(absorptive[ik].min())
    vmax = float(absorptive[ik].max())
    ncols = min(4, len(t2s))
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.6),
                             constrained_layout=True, squeeze=False)
    for it in range(ncols):
        image = axes[0, it].imshow(absorptive[ik, it], origin="lower",
            extent=extent, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, it].set_title(rf"$t_2={t2s[it]:g}$")
        axes[0, it].set_xlabel(r"$\omega_1$")
        axes[0, it].set_ylabel(r"$\omega_3$")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.84,
                 label=r"$S_{\rm abs}^{(3)}$")
    fig.suptitle(rf"Three-level absorptive maps: $\kappa={kappas[ik]:g}$")
    fig.savefig(out / "figure_three_level_waiting_time.png", dpi=180)
    # Transfer-rate effect at the longest waiting time.
    it = len(t2s) - 1
    delta = absorptive[:, it] - absorptive[0, it]
    dmax = float(np.max(abs(delta)))
    fig, axes = plt.subplots(1, len(kappas), figsize=(4 * len(kappas), 3.6),
                             constrained_layout=True, squeeze=False)
    for ik, kappa in enumerate(kappas):
        image = axes[0, ik].imshow(delta[ik], origin="lower", extent=extent,
            aspect="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[0, ik].set_title(rf"$\kappa={kappa:g}$")
        axes[0, ik].set_xlabel(r"$\omega_1$")
        axes[0, ik].set_ylabel(r"$\omega_3$")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.84,
                 label=r"$\Delta S_{\rm abs}^{(3)}$")
    fig.suptitle(rf"Transfer-rate deformation at $t_2={t2s[it]:g}$")
    fig.savefig(out / "figure_three_level_transfer_effect.png", dpi=180)
    print(f"Computed {len(kappas) * len(t2s)} 2D map pairs on "
          f"{args.n_omega}x{args.n_omega}; max pointwise error "
          f"{summary['maximum_pointwise_difference']:.3e}")


if __name__ == "__main__":
    main()
