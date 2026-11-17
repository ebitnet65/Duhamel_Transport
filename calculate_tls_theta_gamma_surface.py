"""Finite-difference spectra and state-generator geometry on a TLS surface."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import calculate_two_control_spectroscopy as tls
import phase_matched_response as response


THETAS = np.array([0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
GAMMA_SCALED = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
T2S = np.array([0.2, 1.0, 5.0])
H_FD = (0.002, 0.02)  # coordinates (theta, gamma_phi / 0.1)
HESSIAN_FD = (0.002, 0.02)


def absorptive_map(theta, gamma_scaled, t2, omegas):
    gamma_phi = tls.GPHI_REF * gamma_scaled
    rp, _, _ = tls.sector_response_map(theta, gamma_phi, t2, omegas, "RP")
    nrp, _, _ = tls.sector_response_map(theta, gamma_phi, t2, omegas, "NRP")
    return -(rp + nrp).real, rp, nrp


def integrated_l2(field, omegas):
    return np.sqrt(np.trapezoid(np.trapezoid(abs(field) ** 2,
                                             omegas, axis=-1),
                                omegas, axis=-1))


def central_derivatives(q, t2s, omegas, h=H_FD):
    q = np.asarray(q, float)
    h0, h1 = h
    e0 = np.array([h0, 0.0])
    e1 = np.array([0.0, h1])
    deriv = np.empty((2, len(t2s), len(omegas), len(omegas)), float)
    deriv_rp = np.empty(deriv.shape, complex)
    deriv_nrp = np.empty(deriv.shape, complex)
    for i, e in enumerate((e0, e1)):
        plus = [absorptive_map(*(q + e), t, omegas) for t in t2s]
        minus = [absorptive_map(*(q - e), t, omegas) for t in t2s]
        for component, result in ((0, deriv), (1, deriv_rp), (2, deriv_nrp)):
            result[i] = (np.asarray([value[component] for value in plus])
                         - np.asarray([value[component] for value in minus])) / (2.0 * h[i])
    return deriv, deriv_rp, deriv_nrp


def second_derivatives(q, t2s, omegas, h=HESSIAN_FD):
    q = np.asarray(q, float)
    h0, h1 = h
    e0 = np.array([h0, 0.0])
    e1 = np.array([0.0, h1])
    f0 = np.asarray([absorptive_map(*q, t, omegas)[0] for t in t2s])
    fp0 = np.asarray([absorptive_map(*(q + e0), t, omegas)[0] for t in t2s])
    fm0 = np.asarray([absorptive_map(*(q - e0), t, omegas)[0] for t in t2s])
    fp1 = np.asarray([absorptive_map(*(q + e1), t, omegas)[0] for t in t2s])
    fm1 = np.asarray([absorptive_map(*(q - e1), t, omegas)[0] for t in t2s])
    pp = np.asarray([absorptive_map(*(q + e0 + e1), t, omegas)[0] for t in t2s])
    pm = np.asarray([absorptive_map(*(q + e0 - e1), t, omegas)[0] for t in t2s])
    mp = np.asarray([absorptive_map(*(q - e0 + e1), t, omegas)[0] for t in t2s])
    mm = np.asarray([absorptive_map(*(q - e0 - e1), t, omegas)[0] for t in t2s])
    return np.stack(((fp0 - 2.0 * f0 + fm0) / h0**2,
                     (pp - pm - mp + mm) / (4.0 * h0 * h1),
                     (fp1 - 2.0 * f0 + fm1) / h1**2))


def christoffel(q, h=(1e-4, 1e-4)):
    J, H = tls.derivs_embedding(q, h)
    metric = J.T @ J
    inv = np.linalg.inv(metric)
    gamma = np.empty((2, 2, 2))
    for mu in range(2):
        for nu in range(2):
            hess = H[0] if (mu, nu) == (0, 0) else H[2] if (mu, nu) == (1, 1) else H[1]
            gamma[:, mu, nu] = inv @ (J.T @ hess)
    return gamma, metric


def transport_segment(Y, qa, qb, nsteps):
    delta = np.asarray(qb, float) - np.asarray(qa, float)
    dt = 1.0 / nsteps

    def rhs(s, value):
        q = np.asarray(qa, float) + s * delta
        conn, _ = christoffel(q)
        A = -np.einsum("amb,m->ab", conn, delta)
        return A @ value

    for step in range(nsteps):
        s = step * dt
        k1 = rhs(s, Y)
        k2 = rhs(s + dt / 2.0, Y + dt * k1 / 2.0)
        k3 = rhs(s + dt / 2.0, Y + dt * k2 / 2.0)
        k4 = rhs(s + dt, Y + dt * k3)
        Y = Y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return Y


def loop_transport(corners, nsteps):
    Y = np.eye(2)
    for qa, qb in zip(corners[:-1], corners[1:]):
        Y = transport_segment(Y, qa, qb, nsteps)
    return Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-omega", type=int, default=51)
    parser.add_argument("--out", default="tls_theta_gamma_surface")
    parser.add_argument("--loop-steps", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    omegas = np.linspace(0.25, 1.75, args.n_omega)
    q0 = np.array([np.pi / 4.0, 1.0])
    t_start = time.time()

    shape = (len(GAMMA_SCALED), len(THETAS), len(T2S),
             len(omegas), len(omegas))
    rp = np.empty(shape, complex)
    nrp = np.empty_like(rp)
    absorptive = np.empty(shape, float)
    for ig, gamma_scaled in enumerate(GAMMA_SCALED):
        for it, theta in enumerate(THETAS):
            for jt, t2 in enumerate(T2S):
                absorptive[ig, it, jt], rp[ig, it, jt], nrp[ig, it, jt] = \
                    absorptive_map(theta, gamma_scaled, t2, omegas)

    # Pointwise convention check at the reference node for both sectors.
    ref_it = int(np.flatnonzero(THETAS == np.pi / 4)[0])
    ref_ig = int(np.flatnonzero(GAMMA_SCALED == 1.0)[0])
    ref_jt = int(np.flatnonzero(T2S == 1.0)[0])
    iw1, iw3 = len(omegas) // 3, 2 * len(omegas) // 3
    pointwise_checks = []
    for sector, maps in (("RP", rp), ("NRP", nrp)):
        direct = response.signal_point_sector(
            np.pi / 4, -float(omegas[iw1]), -float(omegas[iw3]), sector,
            tau2=1.0, w0=tls.W0, gamma_phi=tls.GPHI_REF,
            gamma1=tls.GAMMA1, eta=tls.ETA, detection="full",
            initial="fixed_steady0")
        pointwise_checks.append(dict(
            sector=sector,
            absolute_difference=float(abs(maps[ref_ig, ref_it, ref_jt, iw1, iw3] - direct))))

    fd_first = np.empty((len(GAMMA_SCALED), len(THETAS), 2,
                         len(T2S), len(omegas), len(omegas)), float)
    fd_first_rp = np.empty(fd_first.shape, complex)
    fd_first_nrp = np.empty(fd_first.shape, complex)
    geometry = []
    for ig, gamma_scaled in enumerate(GAMMA_SCALED):
        for it, theta in enumerate(THETAS):
            q = np.array([theta, gamma_scaled])
            fd_first[ig, it], fd_first_rp[ig, it], fd_first_nrp[ig, it] = \
                central_derivatives(q, T2S, omegas)
            geometry.append(tls.geometry_point(theta,
                tls.GPHI_REF * gamma_scaled, h=(1e-4, 1e-4)))

    # Two-coordinate, second-order finite-difference reconstruction about
    # (theta,gamma_phi)=(pi/4,0.1), compared against every exact surface map.
    iq_t = int(np.flatnonzero(THETAS == q0[0])[0])
    iq_g = int(np.flatnonzero(GAMMA_SCALED == q0[1])[0])
    S0 = absorptive[iq_g, iq_t]
    dS, dS_rp, dS_nrp = central_derivatives(q0, T2S, omegas)
    ddS = second_derivatives(q0, T2S, omegas)
    reconstruction = np.empty_like(absorptive)
    reconstruction_errors = []
    for ig, gamma_scaled in enumerate(GAMMA_SCALED):
        for it, theta in enumerate(THETAS):
            delta = np.array([theta, gamma_scaled]) - q0
            pred = (S0 + delta[0] * dS[0] + delta[1] * dS[1]
                    + 0.5 * delta[0]**2 * ddS[0]
                    + delta[0] * delta[1] * ddS[1]
                    + 0.5 * delta[1]**2 * ddS[2])
            reconstruction[ig, it] = pred
            for jt, t2 in enumerate(T2S):
                error = pred[jt] - absorptive[ig, it, jt]
                reconstruction_errors.append(dict(
                    theta=float(theta), gamma_phi=float(tls.GPHI_REF * gamma_scaled),
                    t2=float(t2), relative_integrated_l2_error=float(
                        integrated_l2(error, omegas) /
                        max(integrated_l2(absorptive[ig, it, jt], omegas), 1e-30)),
                    max_absolute_error=float(np.max(abs(error)))))

    # State-generator Levi-Civita holonomy around a small rectangular loop.
    # The transported object is the metric-raised spectral sensitivity map.
    q_a = np.array([q0[0] + 0.70, q0[1]])
    q_b = np.array([q0[0] + 0.70, q0[1] + 0.90])
    q_c = np.array([q0[0], q0[1] + 0.90])
    loop = [q0, q_a, q_b, q_c, q0]
    H_fwd = loop_transport(loop, args.loop_steps)
    H_rev = loop_transport([q0, q_c, q_b, q_a, q0], args.loop_steps)
    gamma0, metric0 = christoffel(q0)
    metric_preservation_error = float(np.linalg.norm(H_fwd.T @ metric0 @ H_fwd - metric0))
    inverse_loop_error = float(np.linalg.norm(H_rev @ H_fwd - np.eye(2)))
    grad_spectrum = np.einsum("ab,btyx->atyx", np.linalg.inv(metric0), dS)
    grad_rp = np.einsum("ab,btyx->atyx", np.linalg.inv(metric0), dS_rp)
    grad_nrp = np.einsum("ab,btyx->atyx", np.linalg.inv(metric0), dS_nrp)
    transported_grad = np.einsum("ab,btyx->atyx", H_fwd, grad_spectrum)
    transported_grad_rp = np.einsum("ab,btyx->atyx", H_fwd, grad_rp)
    transported_grad_nrp = np.einsum("ab,btyx->atyx", H_fwd, grad_nrp)
    delta_grad = transported_grad - grad_spectrum
    delta_grad_rp = transported_grad_rp - grad_rp
    delta_grad_nrp = transported_grad_nrp - grad_nrp
    sensitivity_transport = []

    def metric_field_norm(field):
        density = np.real(np.einsum("axy,ab,bxy->xy", field.conj(), metric0, field))
        return float(np.sqrt(np.trapezoid(np.trapezoid(density, omegas, axis=-1),
                                          omegas, axis=-1)))

    for jt, t2 in enumerate(T2S):
        ratios = {}
        for name, difference, gradient in (
            ("absorptive", delta_grad[:, jt], grad_spectrum[:, jt]),
            ("rephasing", delta_grad_rp[:, jt], grad_rp[:, jt]),
            ("nonrephasing", delta_grad_nrp[:, jt], grad_nrp[:, jt]),
        ):
            ratios[name] = metric_field_norm(difference) / max(metric_field_norm(gradient), 1e-30)
        sensitivity_transport.append(dict(
            t2=float(t2), relative_metric_norm=ratios))

    # FD step check at the base model.
    step_check = []
    derivatives_by_step = []
    for hs in ((0.001, 0.01), (0.002, 0.02), (0.004, 0.04)):
        derivatives_by_step.append(central_derivatives(q0, T2S, omegas, hs)[0])
    for i, hs in enumerate(((0.001, 0.01), (0.002, 0.02), (0.004, 0.04))):
        rel = []
        for jt in range(len(T2S)):
            delta = derivatives_by_step[i][:, jt] - derivatives_by_step[0][:, jt]
            rel.append(float(np.linalg.norm(integrated_l2(delta, omegas)) /
                             max(np.linalg.norm(integrated_l2(
                                 derivatives_by_step[0][:, jt], omegas)), 1e-30)))
        step_check.append(dict(step=list(hs), relative_derivative_change_to_smallest=rel))

    convergence = []
    for n in (max(4, args.loop_steps // 2), args.loop_steps, args.loop_steps * 2):
        h = loop_transport(loop, n)
        convergence.append(dict(steps_per_edge=n, holonomy=h.tolist(),
                               difference_from_finest=None))
    finest = np.asarray(convergence[-1]["holonomy"])
    for item in convergence:
        item["difference_from_finest"] = float(np.linalg.norm(np.asarray(item["holonomy"]) - finest))

    ref_geometry = tls.geometry_point(q0[0], tls.GPHI_REF, h=(1e-4, 1e-4))
    summary = dict(
        model="rotated-dephasing TLS",
        controls=["theta", "gamma_phi/gamma_phi_reference"],
        gamma_phi_reference=tls.GPHI_REF,
        surface=dict(thetas=THETAS.tolist(), gamma_phi_scaled=GAMMA_SCALED.tolist(),
                     gamma_phis=(tls.GPHI_REF * GAMMA_SCALED).tolist(),
                     waiting_times=T2S.tolist(), frequency_range=[0.25, 1.75],
                     n_omega=args.n_omega),
        spectral_convention="positive display axes use signed Fourier arguments -omega1,-omega3; absorptive=-Re(RP+NRP)",
        fd=dict(method="central finite differences of exact RP, NRP, and absorptive spectra",
                steps=H_FD, expansion_center=q0.tolist(),
                second_order_surface_errors=reconstruction_errors,
                step_convergence=step_check),
        pointwise_checks=pointwise_checks,
        state_generator_geometry=dict(
            embedding="real/imag stationary state plus full Liouvillian",
            generator_scale=1.0,
            reference=ref_geometry,
            surface=geometry,
            loop_coordinates=[q.tolist() for q in loop],
            loop_connection="Levi-Civita connection of the induced state-generator metric",
            forward_holonomy=H_fwd.tolist(), reverse_holonomy=H_rev.tolist(),
            metric_preservation_error=metric_preservation_error,
            reverse_forward_composition_error=inverse_loop_error,
            holonomy_convergence=convergence,
            spectral_sensitivity_transport=sensitivity_transport,
            sensitivity_map_definition="metric-raised gradients of RP, NRP, and absorptive spectra, parallel-transported around the loop"),
        elapsed_s=time.time() - t_start,
    )
    np.savez_compressed(out / "tls_theta_gamma_surface.npz",
        omegas=omegas, thetas=THETAS, gamma_phi_scaled=GAMMA_SCALED,
        gamma_phis=tls.GPHI_REF * GAMMA_SCALED, t2s=T2S,
        rephasing=rp, nonrephasing=nrp, absorptive=absorptive,
        fd_first_absorptive=fd_first, fd_second_order_reconstruction=reconstruction,
        fd_first_rephasing=fd_first_rp, fd_first_nonrephasing=fd_first_nrp,
        reference_fd_first=dS, reference_fd_second=ddS,
        reference_fd_first_rephasing=dS_rp,
        reference_fd_first_nonrephasing=dS_nrp,
        reference_spectral_sensitivity=grad_spectrum,
        rephasing_spectral_sensitivity=grad_rp,
        nonrephasing_spectral_sensitivity=grad_nrp,
        transported_spectral_sensitivity=transported_grad,
        transported_rephasing_sensitivity=transported_grad_rp,
        transported_nonrephasing_sensitivity=transported_grad_nrp,
        sensitivity_holonomy_delta=delta_grad,
        rephasing_sensitivity_holonomy_delta=delta_grad_rp,
        nonrephasing_sensitivity_holonomy_delta=delta_grad_nrp,
        state_generator_holonomy=H_fwd, reverse_holonomy=H_rev,
        reference_metric=metric0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [omegas[0], omegas[-1], omegas[0], omegas[-1]]
    t_idx = int(np.flatnonzero(T2S == 1.0)[0])
    gamma_idx = int(np.flatnonzero(GAMMA_SCALED == 1.0)[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, theta in zip(axes, (0.0, np.pi / 4, np.pi / 2)):
        it = int(np.argmin(abs(THETAS - theta)))
        im = ax.imshow(absorptive[gamma_idx, it, t_idx].T,
                       origin="lower", extent=extent, aspect="auto", cmap="magma")
        ax.set_title(rf"$\theta={THETAS[it]:.3f}$")
        ax.set_xlabel(r"$\omega_1$")
        ax.set_ylabel(r"$\omega_3$")
    fig.colorbar(im, ax=axes.tolist(), shrink=.85, label=r"$S_{\rm abs}^{(3)}$")
    fig.suptitle(r"TLS absorptive maps on the $(\theta,\gamma_\phi)$ surface")
    fig.savefig(out / "figure_tls_theta_gamma_maps.png", dpi=180)

    delta_map = np.sqrt(np.maximum(0.0,
        np.einsum("axy,ab,bxy->xy", delta_grad[:, t_idx], metric0,
                  delta_grad[:, t_idx])))
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(delta_map.T, origin="lower", extent=extent,
                   aspect="auto", cmap="magma")
    ax.set_xlabel(r"$\omega_1$")
    ax.set_ylabel(r"$\omega_3$")
    ax.set_title(r"Transported spectral-sensitivity change, $t_2=1$")
    fig.colorbar(im, ax=ax, label=r"$\|\Delta\nabla S_{\rm abs}^{(3)}\|_g$")
    fig.savefig(out / "figure_spectral_sensitivity_holonomy.png", dpi=180)

    curv = np.asarray([g["gaussian_curvature"] for g in geometry]).reshape(
        len(GAMMA_SCALED), len(THETAS))
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(curv, origin="lower", aspect="auto",
        extent=[THETAS[0], THETAS[-1], GAMMA_SCALED[0], GAMMA_SCALED[-1]],
        cmap="RdBu_r")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\gamma_\phi/0.1$")
    ax.set_title("State-generator Gaussian curvature")
    fig.colorbar(im, ax=ax, label="K")
    fig.savefig(out / "figure_state_generator_curvature.png", dpi=180)
    print(f"surface nodes={len(THETAS)*len(GAMMA_SCALED)}, maps="
          f"{len(THETAS)*len(GAMMA_SCALED)*len(T2S)}, "
          f"loop holonomy={H_fwd.tolist()}, elapsed={time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
