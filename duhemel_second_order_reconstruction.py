# duhamel_second_order_reconstruction.py
# Numerical test of first- and second-order Duhamel/resolvent reconstruction
# for a rotated-dephasing two-level system.
#
# Requires:
#   numpy
#   scipy
#   matplotlib
#
# Output figures:
#   duhamel_second_order_error_scaling.png
#   duhamel_second_order_real_theta_0p4.png
#   duhamel_second_order_imag_theta_0p4.png
#   duhamel_second_order_real_theta_0p785.png
#   duhamel_second_order_imag_theta_0p785.png
#   duhamel_second_order_real_theta_1p571.png
#   duhamel_second_order_imag_theta_1p571.png

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from numpy import kron


# ---------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

sig_x = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

sig_z = np.array(
    [
        [-1, 0],
        [0, 1],
    ],
    dtype=complex,
)

sig_m = np.array(
    [
        [0, 1],
        [0, 0],
    ],
    dtype=complex,
)

sig_p = np.array(
    [
        [0, 0],
        [1, 0],
    ],
    dtype=complex,
)

mu = sig_p + sig_m


# ---------------------------------------------------------------------
# Superoperators
# ---------------------------------------------------------------------

def comm_super(O):
    """Return the commutator superoperator vec([O,rho])."""

    return kron(I2, O) - kron(O.T, I2)


def liouvillian(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02):
    """Column-stacked Lindblad Liouvillian.

    The density matrix is vectorized with order="F".
    The induced Liouville basis is

        |gg>>, |eg>>, |ge>>, |ee>>.

    Parameters
    ----------
    w0 : float
        Transition frequency.
    gamma_phi : float
        Pure-dephasing strength.
    theta : float
        Pointer-basis rotation angle.
    gamma1 : float
        Population relaxation rate |e> -> |g>.
    """

    H = 0.5 * w0 * sig_z

    # Hamiltonian contribution: vec(-i[H,rho]).
    L = -1j * (kron(I2, H) - kron(H.T, I2))

    # Rotated dephasing operator.
    A = np.cos(theta) * sig_z + np.sin(theta) * sig_x
    C = np.sqrt(gamma_phi) * A
    CdC = C.conj().T @ C

    L += (
        kron(C.conj(), C)
        - 0.5 * kron(I2, CdC)
        - 0.5 * kron(CdC.T, I2)
    )

    # Population relaxation.
    if gamma1 > 0:
        C1 = np.sqrt(gamma1) * sig_m
        CdC1 = C1.conj().T @ C1

        L += (
            kron(C1.conj(), C1)
            - 0.5 * kron(I2, CdC1)
            - 0.5 * kron(CdC1.T, I2)
        )

    return L


def dL_dtheta(theta=0.0, w0=1.0, gamma_phi=0.1, gamma1=0.02, h=1.0e-6):
    """Central finite difference for dL/dtheta."""

    return (
        liouvillian(w0, gamma_phi, theta + h, gamma1)
        - liouvillian(w0, gamma_phi, theta - h, gamma1)
    ) / (2.0 * h)


def d2L_dtheta2(theta=0.0, w0=1.0, gamma_phi=0.1, gamma1=0.02, h=1.0e-4):
    """Central finite difference for d^2L/dtheta^2."""

    return (
        liouvillian(w0, gamma_phi, theta + h, gamma1)
        - 2.0 * liouvillian(w0, gamma_phi, theta, gamma1)
        + liouvillian(w0, gamma_phi, theta - h, gamma1)
    ) / (h * h)


def G_freq(L, omega, eta=1.0e-3):
    """Frequency-domain propagator.

    G(omega) = - (L + i omega - eta)^(-1).
    """

    return -np.linalg.inv(L + (1j * omega - eta) * np.eye(L.shape[0]))


def U_derivatives(L0, V, W, tau):
    """Return U0, U1, and U2 for U(theta)=exp[L(theta) tau].

    The expansion convention is

        U(theta) = U0 + theta U1 + theta^2 U2/2 + ...

    where

        U1 = dU/dtheta at theta=0
        U2 = d^2U/dtheta^2 at theta=0.

    This uses the block-matrix exponential identity for
    derivatives of a matrix exponential.
    """

    n = L0.shape[0]
    Z = np.zeros_like(L0)

    B = np.block(
        [
            [L0, V, W],
            [Z, L0, 2.0 * V],
            [Z, Z, L0],
        ]
    )

    E = scipy.linalg.expm(B * tau)

    U0 = E[:n, :n]
    U1 = E[:n, n : 2 * n]
    U2 = E[:n, 2 * n : 3 * n]

    return U0, U1, U2


# ---------------------------------------------------------------------
# Model and cached zeroth-order quantities
# ---------------------------------------------------------------------

w0 = 1.0
gamma_phi = 0.1
gamma1 = 0.02
tau2 = 1.0
eta = 1.0e-3

omegas = np.linspace(0.25, 1.75, 241)
theta_grid = np.linspace(0.0, np.pi/2, 31)

J = comm_super(mu)
rho_init = np.array(
    [
        [1, 0],
        [0, 0],
    ],
    dtype=complex,
).reshape(-1, order="F")

# Tr[mu rho] = vec(mu^T)^T vec(rho)
detector = mu.T.reshape(-1, order="F")

L0 = liouvillian(w0, gamma_phi, 0.0, gamma1)
V = dL_dtheta(0.0, w0, gamma_phi, gamma1)
W = d2L_dtheta2(0.0, w0, gamma_phi, gamma1)

U0, U1, U2 = U_derivatives(L0, V, W, tau2)


# Cache G0, G1, G2 at theta = 0.
G0_list = []
G1_list = []
G2_list = []

for omega in omegas:
    G0 = G_freq(L0, omega, eta)
    G1 = G0 @ V @ G0
    G2 = 2.0 * G0 @ V @ G0 @ V @ G0 + G0 @ W @ G0

    G0_list.append(G0)
    G1_list.append(G1)
    G2_list.append(G2)


def signal_from_vec(rho_vec):
    """Return Tr[mu rho] from column-stacked rho."""

    return detector @ rho_vec


def exact_spectrum(theta):
    """Exact finite-theta diagonal spectrum."""

    L = liouvillian(w0, gamma_phi, theta, gamma1)
    U = scipy.linalg.expm(L * tau2)

    S = []

    for omega in omegas:
        G = G_freq(L, omega, eta)
        rho_vec = G @ J @ U @ J @ G @ J @ rho_init
        S.append(signal_from_vec(rho_vec))

    return np.array(S)


def reconstruct_spectrum(theta, order):
    """Duhamel/resolvent reconstruction.

    Parameters
    ----------
    theta : float
        Rotation angle.
    order : int
        0, 1, or 2.
    """

    S = []

    for G0, G1, G2 in zip(G0_list, G1_list, G2_list):

        rho0 = G0 @ J @ U0 @ J @ G0 @ J @ rho_init
        rho = rho0.copy()

        if order >= 1:
            rho1 = (
                G1 @ J @ U0 @ J @ G0 @ J @ rho_init
                + G0 @ J @ U1 @ J @ G0 @ J @ rho_init
                + G0 @ J @ U0 @ J @ G1 @ J @ rho_init
            )

            rho = rho + theta * rho1

        if order >= 2:
            rho2 = (
                G2 @ J @ U0 @ J @ G0 @ J @ rho_init
                + G0 @ J @ U2 @ J @ G0 @ J @ rho_init
                + G0 @ J @ U0 @ J @ G2 @ J @ rho_init
                + 2.0 * G1 @ J @ U1 @ J @ G0 @ J @ rho_init
                + 2.0 * G1 @ J @ U0 @ J @ G1 @ J @ rho_init
                + 2.0 * G0 @ J @ U1 @ J @ G1 @ J @ rho_init
            )

            rho = rho + 0.5 * theta * theta * rho2

        S.append(signal_from_vec(rho))

    return np.array(S)


# ---------------------------------------------------------------------
# Error scan
# ---------------------------------------------------------------------

S0_exact = exact_spectrum(0.0)

err1 = []
err2 = []
deform = []

for theta in theta_grid:
    S_exact = exact_spectrum(theta)
    S_first = reconstruct_spectrum(theta, order=1)
    S_second = reconstruct_spectrum(theta, order=2)

    err1.append(np.linalg.norm(S_first - S_exact) / np.linalg.norm(S_exact))
    err2.append(np.linalg.norm(S_second - S_exact) / np.linalg.norm(S_exact))
    deform.append(np.linalg.norm(S_exact - S0_exact) / np.linalg.norm(S0_exact))

err1 = np.array(err1)
err2 = np.array(err2)
deform = np.array(deform)

print("theta, first_order_error, second_order_error, deformation")
for theta, e1, e2, df in zip(theta_grid, err1, err2, deform):
    print(f"{theta:0.4f}  {e1:0.6f}  {e2:0.6f}  {df:0.6f}")


# ---------------------------------------------------------------------
# Plot error scaling
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(theta_grid, deform, marker="^", label="0th order")
plt.plot(theta_grid, err1, marker="o", label="1st order")
plt.plot(theta_grid, err2, marker="s", label="2nd order")

plt.axhline(0.05, linestyle="--", linewidth=1, label="5% error")
plt.axhline(0.10, linestyle=":", linewidth=1, label="10% error")
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\epsilon^{(2)}$')
#plt.title("First- and second-order reconstruction errors")
plt.legend()
plt.tight_layout()
plt.savefig("duhamel_second_order_error_scaling.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------
# Representative line-shape comparisons
# ---------------------------------------------------------------------

for theta in [0.4, np.pi / 4.0, np.pi / 2.0]:

    S_exact = exact_spectrum(theta)
    S_first = reconstruct_spectrum(theta, order=1)
    S_second = reconstruct_spectrum(theta, order=2)

    tag = str(round(float(theta), 3)).replace(".", "p")

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, np.real(S_exact), label=f"exact theta={theta:.3f}")
    plt.plot(omegas, np.real(S_first), "--", label="1st order")
    plt.plot(omegas, np.real(S_second), ":", label="2nd order")
    plt.xlabel(r'$\omega = \omega_1 = \omega_3$')
    plt.ylabel(r'$\rm{Re}(S^{(3)})$')
    #plt.title(f"Real reconstruction at theta={theta:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"duhamel_second_order_real_theta_{tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, np.imag(S_exact), label=f"exact theta={theta:.3f}")
    plt.plot(omegas, np.imag(S_first), "--", label="1st order")
    plt.plot(omegas, np.imag(S_second), ":", label="2nd order")
    plt.xlabel(r'$\omega = \omega_1 = \omega_3$')
    plt.ylabel(r'$\rm{Im}(S^{(3)})$')
    #plt.title(f"Imaginary reconstruction at theta={theta:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"duhamel_second_order_imag_theta_{tag}.png", dpi=200)
    plt.show()
