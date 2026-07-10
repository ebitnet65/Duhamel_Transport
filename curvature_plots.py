# curvature_plots.py
# Generates Liouvillian curvature tensor plots for the rotated-dephasing 2LS.
#
# Requires:
#   numpy
#   matplotlib
#
# Output files:
#   curvature_real_elements_theta.png
#   curvature_imag_elements_theta.png
#   curvature_norm_theta_dense.png
#   curvature_Re_theta0.png
#   curvature_Im_theta0.png
#   curvature_Re_thetapi2.png
#   curvature_Im_thetapi2.png

import numpy as np
import matplotlib.pyplot as plt
from numpy import kron


# ---------------------------------------------------------------------
# Basic two-level-system operators
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

sig_p = np.array(
    [
        [0, 0],
        [1, 0],
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


# ---------------------------------------------------------------------
# Liouvillian and numerical derivatives
# ---------------------------------------------------------------------

def liouvillian(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02):
    """Column-stacked Lindblad Liouvillian.

    The vectorization convention is order="F". For a 2x2 density matrix,
    the induced basis ordering is

        |gg>>, |eg>>, |ge>>, |ee>>.

    Parameters
    ----------
    w0 : float
        Transition frequency.
    gamma_phi : float
        Pure-dephasing strength.
    theta : float
        Rotation angle of the environmental pointer axis.
    gamma1 : float
        Population-relaxation rate for |e> -> |g>.
    """

    H = 0.5 * w0 * sig_z

    # Hamiltonian part: vec(-i[H,rho]).
    L = -1j * (kron(I2, H) - kron(H.T, I2))

    # Rotated dephasing axis.
    A = np.cos(theta) * sig_z + np.sin(theta) * sig_x
    C = np.sqrt(gamma_phi) * A
    CdC = C.conj().T @ C

    L += (
        kron(C.conj(), C)
        - 0.5 * kron(I2, CdC)
        - 0.5 * kron(CdC.T, I2)
    )

    # Population relaxation |e> -> |g>.
    if gamma1 > 0:
        C1 = np.sqrt(gamma1) * sig_m
        CdC1 = C1.conj().T @ C1
        L += (
            kron(C1.conj(), C1)
            - 0.5 * kron(I2, CdC1)
            - 0.5 * kron(CdC1.T, I2)
        )

    return L


def dL_dtheta(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02, h=1.0e-5):
    """Central finite-difference derivative dL/dtheta."""

    return (
        liouvillian(w0, gamma_phi, theta + h, gamma1)
        - liouvillian(w0, gamma_phi, theta - h, gamma1)
    ) / (2.0 * h)


def dL_dgamma(w0=1.0, gamma_phi=0.1, theta=0.0, gamma1=0.02, h=1.0e-5):
    """Central finite-difference derivative dL/dgamma_phi."""

    return (
        liouvillian(w0, gamma_phi + h, theta, gamma1)
        - liouvillian(w0, gamma_phi - h, theta, gamma1)
    ) / (2.0 * h)


def curvature_matrix(
    theta,
    w0=1.0,
    gamma_phi=0.1,
    gamma1=0.02,
    g_beta=-0.03,
    theta_beta=-0.05,
):
    """Compute F_{t beta} = [L, d_beta L].

    The beta derivative is modeled as

        d_beta L =
            g_beta d_{gamma_phi} L + theta_beta d_theta L.

    """

    L = liouvillian(
        w0=w0,
        gamma_phi=gamma_phi,
        theta=theta,
        gamma1=gamma1,
    )

    dLbeta = (
        g_beta * dL_dgamma(w0, gamma_phi, theta, gamma1)
        + theta_beta * dL_dtheta(w0, gamma_phi, theta, gamma1)
    )

    return L @ dLbeta - dLbeta @ L


# ---------------------------------------------------------------------
# Sweep over theta
# ---------------------------------------------------------------------

w0 = 1.0
gamma_phi = 0.1
gamma1 = 0.02
g_beta = -0.03
theta_beta = -0.05

theta_grid = np.linspace(0.0, np.pi/2, 301)

F_grid = np.array(
    [
        curvature_matrix(
            th,
            w0=w0,
            gamma_phi=gamma_phi,
            gamma1=gamma1,
            g_beta=g_beta,
            theta_beta=theta_beta,
        )
        for th in theta_grid
    ]
)

real_max = np.max(np.abs(np.real(F_grid)))
imag_max = np.max(np.abs(np.imag(F_grid)))
fro_norm = np.linalg.norm(F_grid.reshape(len(theta_grid), -1), axis=1)

print("max abs Re(F) =", real_max)
print("max abs Im(F) =", imag_max)
print("F(theta = 0):")
print(F_grid[0])
print("F(theta = pi/2):")
print(F_grid[len(theta_grid) // 2])


# ---------------------------------------------------------------------
# Plot matrix elements as functions of theta
# ---------------------------------------------------------------------

threshold = 1.0e-6
nonzero_real = []
nonzero_imag = []

for i in range(4):
    for j in range(4):
        if np.max(np.abs(np.real(F_grid[:, i, j]))) > threshold:
            nonzero_real.append((i, j))
        if np.max(np.abs(np.imag(F_grid[:, i, j]))) > threshold:
            nonzero_imag.append((i, j))

basis_labels = ["gg", "eg", "ge", "ee"]

plt.figure(figsize=(8, 5))
for i, j in nonzero_real:
    plt.plot(
        theta_grid,
        np.real(F_grid[:, i, j]),
        label=f"Re F[{basis_labels[i]},{basis_labels[j]}]",
    )
plt.xlabel("theta")
plt.ylabel("Real curvature matrix element")
plt.title("Real parts of nonzero Liouvillian curvature elements")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("curvature_real_elements_theta.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
for i, j in nonzero_imag:
    plt.plot(
        theta_grid,
        np.imag(F_grid[:, i, j]),
        label=f"Im F[{basis_labels[i]},{basis_labels[j]}]",
    )
plt.xlabel("theta")
plt.ylabel("Imaginary curvature matrix element")
plt.title("Imaginary parts of nonzero Liouvillian curvature elements")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("curvature_imag_elements_theta.png", dpi=200)
plt.show()

plt.figure(figsize=(7, 4.5))
plt.plot(theta_grid, fro_norm)
plt.xlabel("theta")
plt.ylabel("Frobenius norm")
plt.title("Liouvillian curvature norm")
plt.tight_layout()
plt.savefig("curvature_norm_theta_dense.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------
# Representative matrix heat maps
# ---------------------------------------------------------------------

def plot_matrix_heatmap(M, title, filename):
    vmax = np.max(np.abs(M))
    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(M, vmin=-vmax, vmax=vmax)
    plt.xticks(range(4), basis_labels)
    plt.yticks(range(4), basis_labels)
    plt.colorbar(label="matrix element")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


F0 = F_grid[0]
Fmid = F_grid[len(theta_grid) // 2]

plot_matrix_heatmap(
    np.real(F0),
    "Re F at theta = 0",
    "curvature_Re_theta0.png",
)

plot_matrix_heatmap(
    np.imag(F0),
    "Im F at theta = 0",
    "curvature_Im_theta0.png",
)

plot_matrix_heatmap(
    np.real(Fmid),
    "Re F at theta = pi/2",
    "curvature_Re_thetapi2.png",
)

plot_matrix_heatmap(
    np.imag(Fmid),
    "Im F at theta = pi/2",
    "curvature_Im_thetapi2.png",
)
