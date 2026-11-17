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
KAPPAS = np.array([0.0, 0.02, 0.03, 0.06])
T2S = np.array([0.2, 1.0, 5.0, 10.0])
FD_CENTER = 0.03
FD_STEP = 0.005
FD_STEPS = np.array([0.0025, 0.005, 0.01])


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

    # Finite-difference the exact response maps with respect to the transfer
    # rate, then build a second-order local reconstruction about kappa=0.02.
    # This is the FD/Duhamel-style comparison; the maps above remain exact.
    fd_cache = {}
    for h in FD_STEPS:
        for it, t2 in enumerate(t2s):
            for sector in ("RP", "NRP"):
                plus, _ = sector_map(float(FD_CENTER + h), float(t2), omegas, sector)
                minus, _ = sector_map(float(FD_CENTER - h), float(t2), omegas, sector)
                fd_cache[(float(h), it, sector)] = (plus, minus)

    first_rp = np.empty((len(t2s), len(omegas), len(omegas)), complex)
    second_rp = np.empty_like(first_rp)
    first_nrp = np.empty_like(first_rp)
    second_nrp = np.empty_like(first_rp)
    for it in range(len(t2s)):
        for sector, first_out, second_out in (
            ("RP", first_rp, second_rp), ("NRP", first_nrp, second_nrp)
        ):
            plus, minus = fd_cache[(FD_STEP, it, sector)]
            center = rp[np.flatnonzero(kappas == FD_CENTER)[0], it] if sector == "RP" else nrp[np.flatnonzero(kappas == FD_CENTER)[0], it]
            first_out[it] = (plus - minus) / (2.0 * FD_STEP)
            second_out[it] = (plus - 2.0 * center + minus) / (FD_STEP ** 2)

    center_index = int(np.flatnonzero(kappas == FD_CENTER)[0])
    center_absorptive = absorptive[center_index]
    first_absorptive = -(first_rp + first_nrp).real
    second_absorptive = -(second_rp + second_nrp).real
    fd_reconstructed = np.empty_like(absorptive)
    fd_records = []
    for ik, kappa in enumerate(kappas):
        delta = float(kappa - FD_CENTER)
        fd_reconstructed[ik] = (center_absorptive + delta * first_absorptive
                                + 0.5 * delta * delta * second_absorptive)
        for it, t2 in enumerate(t2s):
            diff = fd_reconstructed[ik, it] - absorptive[ik, it]
            denom = np.trapezoid(np.trapezoid(abs(absorptive[ik, it]) ** 2,
                                              omegas, axis=0), omegas)
            err = np.trapezoid(np.trapezoid(abs(diff) ** 2, omegas, axis=0), omegas)
            fd_records.append(dict(
                kappa=float(kappa), t2=float(t2), expansion_center=FD_CENTER,
                delta_kappa=delta,
                relative_integrated_l2_error=float(np.sqrt(err / max(denom, 1e-30))),
                max_absolute_error=float(np.max(abs(diff))),
            ))

    fd_step_convergence = []
    for h in FD_STEPS:
        deriv_norms = []
        for it in range(len(t2s)):
            plus_rp, minus_rp = fd_cache[(float(h), it, "RP")]
            plus_nrp, minus_nrp = fd_cache[(float(h), it, "NRP")]
            deriv_norms.append((-(plus_rp + plus_nrp).real
                                + (minus_rp + minus_nrp).real) / (2.0 * h))
        if float(h) == float(FD_STEPS[0]):
            fd_reference_derivative = deriv_norms
        fd_step_convergence.append(dict(
            step=float(h),
            derivative_relative_l2_to_smallest_step=[
                float(np.sqrt(np.trapezoid(np.trapezoid(abs(deriv_norms[it] - fd_reference_derivative[it]) ** 2,
                                                        omegas, axis=0), omegas)
                              / max(np.trapezoid(np.trapezoid(abs(fd_reference_derivative[it]) ** 2,
                                                              omegas, axis=0), omegas), 1e-30)))
                for it in range(len(t2s))
            ] if float(h) != float(FD_STEPS[0]) else [0.0] * len(t2s),
        ))

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
        fd_center_kappa=FD_CENTER, fd_step=FD_STEP,
        fd_first_rp=first_rp, fd_second_rp=second_rp,
        fd_first_nrp=first_nrp, fd_second_nrp=second_nrp,
        fd_first_absorptive=first_absorptive,
        fd_second_absorptive=second_absorptive,
        fd_second_order_reconstruction=fd_reconstructed,
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
        finite_difference=dict(
            method="central finite differences of exact RP and NRP maps versus kappa",
            expansion_center=FD_CENTER, selected_step=FD_STEP,
            tested_steps=FD_STEPS.tolist(),
            second_order_reconstruction_errors=fd_records,
            first_derivative_step_convergence=fd_step_convergence,
        ),
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
    # Compare the exact and FD second-order maps at the longest waiting time.
    t_index = len(t2s) - 1
    target_indices = [i for i, value in enumerate(kappas) if value != FD_CENTER]
    fig, axes = plt.subplots(3, len(target_indices),
                             figsize=(4 * len(target_indices), 9),
                             constrained_layout=True, squeeze=False)
    for col, ik in enumerate(target_indices):
        exact = absorptive[ik, t_index]
        reconstruction = fd_reconstructed[ik, t_index]
        err_map = abs(reconstruction - exact)
        limit = float(max(abs(exact).max(), abs(reconstruction).max()))
        axes[0, col].imshow(exact, origin="lower", extent=extent, aspect="auto",
                            cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[1, col].imshow(reconstruction, origin="lower", extent=extent,
                            aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        im = axes[2, col].imshow(err_map, origin="lower", extent=extent,
                                 aspect="auto", cmap="magma")
        axes[0, col].set_title(rf"Exact, $\kappa={kappas[ik]:g}$")
        axes[1, col].set_title("FD second order")
        axes[2, col].set_title("Absolute map error")
        for row in range(3):
            axes[row, col].set_xlabel(r"$\omega_1$")
            axes[row, col].set_ylabel(r"$\omega_3$")
    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=.82,
                 label=r"$|S_{\rm FD}^{(3)}-S_{\rm exact}^{(3)}|$")
    fig.suptitle(rf"Local FD reconstruction at $t_2={t2s[t_index]:g}$, "
                 rf"center $\kappa={FD_CENTER:g}$")
    fig.savefig(out / "figure_fd_reconstruction.png", dpi=180)
    print(f"Computed {len(kappas) * len(t2s)} 2D map pairs on "
          f"{args.n_omega}x{args.n_omega}; max pointwise error "
          f"{summary['maximum_pointwise_difference']:.3e}")


if __name__ == "__main__":
    main()
