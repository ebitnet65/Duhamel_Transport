import numpy as np

import Liouville_transport_general as general


def ket(index, dim=3):
    out = np.zeros((dim, 1), dtype=complex)
    out[index, 0] = 1.0
    return out


def projector(index, dim=3):
    k = ket(index, dim)
    return k @ k.conj().T


def op(i, j, dim=3):
    return ket(i, dim) @ ket(j, dim).conj().T


def dissipator(collapse):
    dim = collapse.shape[0]
    ident = np.eye(dim, dtype=complex)
    cdc = collapse.conj().T @ collapse
    return (
        np.kron(collapse.conj(), collapse)
        - 0.5 * np.kron(ident, cdc)
        - 0.5 * np.kron(cdc.T, ident)
    )


def v_system_liouvillian(params):
    dim = 3
    ident = np.eye(dim, dtype=complex)
    hamiltonian = (
        params["omega_b"] * projector(1, dim)
        + params["omega_d"] * projector(2, dim)
        + params["v_bd"] * (op(1, 2, dim) + op(2, 1, dim))
    )
    liouvillian = -1j * (
        np.kron(ident, hamiltonian)
        - np.kron(hamiltonian.T, ident)
    )

    gamma_b = params["gamma_b"]
    gamma_d = params["gamma_d"]
    gamma_bd_phi = params["gamma_bd_phi"]

    if gamma_b > 0:
        liouvillian += dissipator(np.sqrt(gamma_b) * op(0, 1, dim))
    if gamma_d > 0:
        liouvillian += dissipator(np.sqrt(gamma_d) * op(0, 2, dim))
    if gamma_bd_phi > 0:
        liouvillian += dissipator(
            np.sqrt(gamma_bd_phi)
            * (projector(1, dim) - projector(2, dim))
        )
    return liouvillian


def v_system_mu_components(params):
    dim = 3
    mu_gb = params["mu_gb"]
    mu_gd = params["mu_gd"]
    mu_plus = mu_gb * op(1, 0, dim) + mu_gd * op(2, 0, dim)
    mu_minus = mu_plus.conj().T
    return mu_plus, mu_minus


def v_system_model():
    return general.LiouvilleModel(
        dimension=3,
        liouvillian=v_system_liouvillian,
        initial_state=lambda _params: general.vec(projector(0, 3)),
        transition_dipole=lambda params: sum(v_system_mu_components(params)),
        dipole_components=v_system_mu_components,
    )


def v_system_parameter_map(*controls):
    if not controls:
        controls = (general.ControlSpec("v_bd", base_value=0.0),)
    return general.ParameterMap(
        base_params={
            "omega_b": 1.0,
            "omega_d": 1.18,
            "v_bd": 0.0,
            "gamma_b": 0.03,
            "gamma_d": 0.004,
            "gamma_bd_phi": 0.02,
            "mu_gb": 1.0,
            "mu_gd": 0.08,
        },
        controls=tuple(controls),
    )


def test_v_system_liouvillian_is_trace_preserving():
    model = v_system_model()
    params = v_system_parameter_map().params([0.05])
    liouvillian = model.L(params)
    trace_row = general.vec(np.eye(3)).conj()

    np.testing.assert_allclose(trace_row @ liouvillian, np.zeros(9), atol=1e-12)


def test_v_system_dipole_components_are_explicit_and_not_triangular_order_dependent():
    model = v_system_model()
    params = v_system_parameter_map().params([0.0])
    mu_plus, mu_minus = model.mu_components(params)

    np.testing.assert_allclose(mu_plus[1, 0], params["mu_gb"])
    np.testing.assert_allclose(mu_plus[2, 0], params["mu_gd"])
    np.testing.assert_allclose(mu_minus[0, 1], params["mu_gb"])
    np.testing.assert_allclose(mu_minus[0, 2], params["mu_gd"])
    np.testing.assert_allclose(model.mu(params), mu_plus + mu_minus)


def test_v_system_general_transport_has_identity_at_zero_control():
    model = v_system_model()
    pmap = v_system_parameter_map()
    omegas = np.linspace(0.85, 1.3, 6)
    candidates = general.canonical_pathways(("RP",))
    pathways, candidate_residuals = general.independent_canonical_pathways(
        model,
        pmap,
        omegas,
        candidates,
        stage="after_U2",
    )

    result = general.liouville_transport_matrix(
        model,
        pmap,
        np.array([0.0]),
        omegas,
        pathways,
        stage="after_U2",
        rcond=1e-11,
    )

    assert len(pathways) == 2
    assert np.count_nonzero(candidate_residuals > 1e-10) == 2
    np.testing.assert_allclose(result.T, np.eye(len(pathways)), atol=1e-9)
    assert result.rank == len(pathways)
    assert result.P0.shape[0] == len(omegas) * 9
    assert result.P0.shape[1] == len(pathways)
    assert np.max(result.residuals) < 1e-9


def test_v_system_can_expand_wrt_subset_of_parameters():
    model = v_system_model()
    pmap = v_system_parameter_map(
        general.ControlSpec("v_bd", base_value=0.0, scale=0.02),
        general.ControlSpec("mu_gd", base_value=0.08, scale=0.01),
    )

    params = pmap.params([2.0, -3.0])
    assert params["v_bd"] == 0.04
    assert params["mu_gd"] == 0.05
    assert params["omega_b"] == 1.0

    derivatives = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)

    assert set(derivatives) == {(), (0,), (1,), (0, 0), (0, 1), (1, 1)}
    assert derivatives[()].shape == (9, 9)
    assert np.linalg.norm(derivatives[(0,)]) > 0.0
    np.testing.assert_allclose(derivatives[(1,)], np.zeros((9, 9)), atol=1e-10)
