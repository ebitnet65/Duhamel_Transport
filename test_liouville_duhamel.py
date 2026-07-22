import numpy as np
import scipy.linalg

from Liouville_duhamel import (
    propagator_duhamel_expansion,
    resolvent_derivative_expansion,
)


def _commuting_derivatives():
    L0 = np.diag([-0.2 + 1.0j, -0.4 + 1.3j])
    L1 = np.diag([0.03 - 0.02j, -0.01 + 0.04j])
    L2 = np.diag([0.005 + 0.01j, 0.02 - 0.003j])
    L12 = np.diag([0.007 - 0.002j, -0.004 + 0.006j])
    return {
        (): L0,
        (0,): L1,
        (1,): L2,
        (0, 0): np.zeros_like(L0),
        (0, 1): L12,
        (1, 1): np.zeros_like(L0),
    }


def test_resolvent_derivative_expansion_matches_finite_difference():
    derivatives = _commuting_derivatives()
    omega = 1.1
    eta = 0.003
    expansion = resolvent_derivative_expansion(derivatives, omega, eta)
    lam = np.array([0.12, -0.08])

    def L_at(lam_vec):
        return (
            derivatives[()]
            + lam_vec[0] * derivatives[(0,)]
            + lam_vec[1] * derivatives[(1,)]
            + lam_vec[0] * lam_vec[1] * derivatives[(0, 1)]
        )

    ident = np.eye(2, dtype=complex)
    exact = -np.linalg.inv(L_at(lam) + (1j * omega - eta) * ident)
    approx = expansion.evaluate(lam, order=2)

    np.testing.assert_allclose(approx, exact, rtol=2e-5, atol=2e-5)


def test_propagator_duhamel_expansion_matches_commuting_analytic_coefficients():
    derivatives = _commuting_derivatives()
    tau = 0.7
    expansion = propagator_duhamel_expansion(derivatives, tau, n_quad=16)

    U0 = scipy.linalg.expm(derivatives[()] * tau)
    expected_first0 = U0 @ (tau * derivatives[(0,)])
    expected_first1 = U0 @ (tau * derivatives[(1,)])
    expected_second01 = U0 @ (
        tau * derivatives[(0, 1)]
        + tau * tau * derivatives[(0,)] @ derivatives[(1,)]
    )

    np.testing.assert_allclose(expansion.value, U0)
    np.testing.assert_allclose(expansion.first[0], expected_first0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(expansion.first[1], expected_first1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        expansion.second[(0, 1)],
        expected_second01,
        rtol=1e-10,
        atol=1e-10,
    )
