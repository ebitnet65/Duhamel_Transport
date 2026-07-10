# Python code used for corrected diagonal theta sweeps and KK check.
# Requires: numpy, scipy, matplotlib

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from numpy import kron
from scipy.signal import hilbert

I2 = np.eye(2, dtype=complex)
sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
sig_z = np.array([[-1, 0], [0, 1]], dtype=complex)
sig_p = np.array([[0, 0], [1, 0]], dtype=complex)
sig_m = np.array([[0, 1], [0, 0]], dtype=complex)
mu = sig_p + sig_m

def comm_super(O):
    return kron(I2, O) - kron(O.T, I2)

def liouvillian(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02):
    H = 0.5 * w0 * sig_z
    L = -1j * (kron(I2, H) - kron(H.T, I2))

    A = np.cos(theta) * sig_z + np.sin(theta) * sig_x
    C = np.sqrt(gamma_phi) * A
    CdC = C.conj().T @ C
    L += kron(C.conj(), C) - 0.5 * kron(I2, CdC) - 0.5 * kron(CdC.T, I2)

    if gamma1 > 0:
        C1 = np.sqrt(gamma1) * sig_m
        CdC1 = C1.conj().T @ C1
        L += kron(C1.conj(), C1) - 0.5 * kron(I2, CdC1) - 0.5 * kron(CdC1.T, I2)

    return L

def G_freq(L, omega, eta=1e-3):
    n = L.shape[0]
    return -np.linalg.inv(L + (1j * omega - eta) * np.eye(n))

def signal_point_diag(theta, omega, tau2=1.0, w0=1.0, gamma_phi=0.1, gamma1=0.02):
    L = liouvillian(w0=w0, gamma_phi=gamma_phi, theta=theta, gamma1=gamma1)
    J = comm_super(mu)
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex).reshape(-1, order="F")
    U2 = scipy.linalg.expm(L * tau2)
    G = G_freq(L, omega)
    rho_vec = G @ J @ U2 @ J @ G @ J @ rho0
    rho = rho_vec.reshape((2, 2), order="F")
    return np.trace(mu @ rho)

def dL_dtheta(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02, h=1e-5):
    return (
        liouvillian(w0, gamma_phi, theta + h, gamma1)
        - liouvillian(w0, gamma_phi, theta - h, gamma1)
    ) / (2 * h)

def dL_dgamma(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02, h=1e-5):
    return (
        liouvillian(w0, gamma_phi + h, theta, gamma1)
        - liouvillian(w0, gamma_phi - h, theta, gamma1)
    ) / (2 * h)

def curvature_norm(theta, w0=1.0, gamma_phi=0.1, gamma1=0.02,
                   g_beta=-0.03, theta_beta=-0.05):
    L = liouvillian(w0=w0, gamma_phi=gamma_phi, theta=theta, gamma1=gamma1)
    dLbeta = (
        g_beta * dL_dgamma(w0, gamma_phi, theta, gamma1)
        + theta_beta * dL_dtheta(w0, gamma_phi, theta, gamma1)
    )
    F = L @ dLbeta - dLbeta @ L
    return np.linalg.norm(F)

# Parameters
w0 = 1.0
omega0 = 1.0
gamma_phi = 0.1
gamma1 = 0.02
tau2 = 1.0

delta = 0.75
omegas = np.linspace(omega0 - delta, omega0 + delta, 1001)

theta_values = np.array([
    0,
    np.pi / 8,
    np.pi / 4,
    3 * np.pi / 8,
    np.pi / 2
])
theta_labels = ["0", r'$\pi/8$', r'$\pi/4$', r'$\pi/8$', r'$\pi/2$']

diag_signals = []
for th in theta_values:
    diag_signals.append(
        np.array([
            signal_point_diag(th, om, tau2, w0, gamma_phi, gamma1)
            for om in omegas
        ])
    )
diag_signals = np.array(diag_signals)
S_ref = diag_signals[0]

diff_integrals = np.array([
    np.trapezoid(np.abs(S - S_ref)**2, omegas)
    for S in diag_signals
])
curv_norms = np.array([
    curvature_norm(th, w0, gamma_phi, gamma1)
    for th in theta_values
])

left = omegas < omega0
right = omegas > omega0

def matched_asymmetry(S, component="abs"):
    Re = np.real(S)
    Im = np.imag(S)
    Abs = np.abs(S)

    x_left = omega0 - omegas[left][::-1]
    x_right = omegas[right] - omega0
    x_common = np.linspace(0, min(x_left[-1], x_right[-1]), 500)

    if component == "real":
        yL = np.interp(x_common, x_left, np.abs(Re[left][::-1]))
        yR = np.interp(x_common, x_right, np.abs(Re[right]))
    elif component == "imag":
        yL = np.interp(x_common, x_left, np.abs(Im[left][::-1]))
        yR = np.interp(x_common, x_right, np.abs(Im[right]))
    else:
        yL = np.interp(x_common, x_left, Abs[left][::-1])
        yR = np.interp(x_common, x_right, Abs[right])

    Lw = np.trapezoid(yL, x_common)
    Rw = np.trapezoid(yR, x_common)
    return (Rw - Lw) / (Rw + Lw)

asym_re = np.array([matched_asymmetry(S, "real") for S in diag_signals])
asym_im = np.array([matched_asymmetry(S, "imag") for S in diag_signals])
asym_abs = np.array([matched_asymmetry(S, "abs") for S in diag_signals])

# Plot corrected real diagonal sweep
plt.figure(figsize=(8, 5))
for S, lab in zip(diag_signals, theta_labels):
    plt.plot(omegas, np.real(S), label=lab)
plt.xlabel(r'$\omega = \omega_1 = \omega_3$')
plt.ylabel(r'$\rm{Re}(S^{(3)}(\omega, \omega))$')
#plt.title("Corrected matched-window real diagonal sweep")
plt.legend(title=r'$\theta$')
plt.tight_layout()
plt.savefig("corrected_diag_theta_real.png", dpi=200)
plt.show()

# Plot corrected imaginary diagonal sweep
plt.figure(figsize=(8, 5))
for S, lab in zip(diag_signals, theta_labels):
    plt.plot(omegas, np.imag(S), label=lab)
plt.xlabel(r'$\omega = \omega_1 = \omega_3$')
plt.ylabel(r'$\rm{Im}(S^{(3)}(\omega, \omega))$')
#plt.title("Corrected matched-window imaginary diagonal sweep")
plt.legend(title=r'$\theta$')
plt.tight_layout()
plt.savefig("corrected_diag_theta_imag.png", dpi=200)
plt.show()

# Real difference from theta = 0
plt.figure(figsize=(8, 5))
for S, lab in zip(diag_signals[1:], theta_labels[1:]):
    plt.plot(omegas, np.real(S - S_ref), label=lab)
plt.xlabel("omega = omega1 = omega3")
plt.ylabel("Delta Re S3(omega, omega)")
plt.title("Corrected real diagonal difference relative to theta = 0")
plt.legend(title="theta")
plt.tight_layout()
plt.savefig("corrected_diag_theta_real_difference.png", dpi=200)
plt.show()

# Imaginary difference from theta = 0
plt.figure(figsize=(8, 5))
for S, lab in zip(diag_signals[1:], theta_labels[1:]):
    plt.plot(omegas, np.imag(S - S_ref), label=lab)
plt.xlabel("omega = omega1 = omega3")
plt.ylabel("Delta Im S3(omega, omega)")
plt.title("Corrected imaginary diagonal difference relative to theta = 0")
plt.legend(title="theta")
plt.tight_layout()
plt.savefig("corrected_diag_theta_imag_difference.png", dpi=200)
plt.show()

# Corrected asymmetry
plt.figure(figsize=(7, 4.5))
plt.plot(theta_values, asym_re, marker="o", label="Real")
plt.plot(theta_values, asym_im, marker="s", label="Imaginary")
plt.plot(theta_values, asym_abs, marker="^", label="Magnitude")
plt.axhline(0, linewidth=1)
plt.xlabel("theta")
plt.ylabel("matched-window side asymmetry")
plt.title("Corrected diagonal asymmetry around omega0")
plt.legend()
plt.tight_layout()
plt.savefig("corrected_diag_theta_asymmetry.png", dpi=200)
plt.show()

# Spectral deformation vs curvature norm
plt.figure(figsize=(7, 4.5))
scaled_curv = curv_norms / np.max(curv_norms) * np.max(diff_integrals)
plt.plot(theta_values, diff_integrals, marker="o", label="integrated |Delta S|^2")
plt.plot(theta_values, scaled_curv, marker="s", label="scaled curvature norm")
plt.xlabel("theta")
plt.ylabel("relative diagnostic")
plt.title("Corrected spectral deformation vs Liouvillian curvature norm")
plt.legend()
plt.tight_layout()
plt.savefig("corrected_theta_diagnostics.png", dpi=200)
plt.show()

# Finite-window Kramers--Kronig diagnostic for theta = 0
S0 = diag_signals[0]
Re0 = np.real(S0)
Im0 = np.imag(S0)

Re0_centered = Re0 - 0.5 * (Re0[0] + Re0[-1])
Im0_centered = Im0 - 0.5 * (Im0[0] + Im0[-1])

H_Re = np.imag(hilbert(Re0_centered))
scale = np.dot(H_Re, Im0_centered) / np.dot(H_Re, H_Re)
kk_resid = Im0_centered - scale * H_Re
rel_kk_error = np.linalg.norm(kk_resid) / np.linalg.norm(Im0_centered)

print("Finite-window KK check, theta = 0")
print("best-fit Hilbert scale:", scale)
print("relative KK residual:", rel_kk_error)

plt.figure(figsize=(7, 4.5))
plt.plot(omegas, Im0_centered, label="Im S3, centered")
plt.plot(omegas, scale * H_Re, linestyle="--",
         label="scaled Hilbert transform of Re")
plt.xlabel("omega = omega1 = omega3")
plt.ylabel("KK comparison")
plt.title("Finite-window KK check for theta = 0 diagonal response")
plt.legend()
plt.tight_layout()
plt.savefig("corrected_theta0_KK_check.png", dpi=200)
plt.show()


#--------------TESTING RP and NRP ROBUSTNESS


