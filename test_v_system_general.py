import numpy as np

import Liouville_transport_general as general
from Liouville_qutip_adapter import qutip_model_from_hamiltonian


def ket(index, dim=3):
    out = np.zeros((dim, 1), dtype=complex)
    out[index, 0] = 1.0
    return out


def projector(index, dim=3):
    k = ket(index, dim)
    return k @ k.conj().T


def op(i, j, dim=3):
    return ket(i, dim) @ ket(j, dim).conj().T


def liouvillian_from_hamiltonian_and_lindblad_ops(hamiltonian, lindblad_ops):
    return general.liouvillian_from_hamiltonian_and_lindblad_ops(
        hamiltonian,
        lindblad_ops,
    )


def detailed_balance_transfer_rates(params):
    """Return incoherent transfer rates for b -> d and d -> b.

    The symmetric prefactor convention enforces

        k_{b->d} / k_{d->b} = exp[-beta (omega_d - omega_b)].

    Users may specify either inverse temperature `beta_bd` directly or a
    positive temperature `temperature` with k_B = 1.
    """

    delta = params["omega_d"] - params["omega_b"]
    if "beta_bd" in params:
        beta = params["beta_bd"]
    else:
        temperature = params["temperature"]
        if temperature <= 0:
            beta = np.inf
        else:
            beta = 1.0 / temperature
    rate = params["k_bd"]
    if np.isinf(beta):
        if delta > 0:
            return 0.0, rate
        if delta < 0:
            return rate, 0.0
        return rate, rate
    return (
        rate * np.exp(-0.5 * beta * delta),
        rate * np.exp(+0.5 * beta * delta),
    )


def v_system_hamiltonian(params):
    dim = 3
    return (
        params["omega_b"] * projector(1, dim)
        + params["omega_d"] * projector(2, dim)
        + params["J_bd"] * (op(1, 2, dim) + op(2, 1, dim))
    )


def v_system_lindblad_ops(params):
    """Return the Hilbert-space collapse operators C_k for the V-system."""

    dim = 3
    collapse_ops = []

    gamma_b = params["gamma_b"]
    gamma_d = params["gamma_d"]
    gamma_phi_b = params["gamma_phi_b"]
    gamma_phi_d = params["gamma_phi_d"]
    k_b_to_d, k_d_to_b = detailed_balance_transfer_rates(params)

    if gamma_b > 0:
        collapse_ops.append(np.sqrt(gamma_b) * op(0, 1, dim))
    if gamma_d > 0:
        collapse_ops.append(np.sqrt(gamma_d) * op(0, 2, dim))
    if gamma_phi_b > 0:
        collapse_ops.append(np.sqrt(gamma_phi_b) * projector(1, dim))
    if gamma_phi_d > 0:
        collapse_ops.append(np.sqrt(gamma_phi_d) * projector(2, dim))
    if k_b_to_d > 0:
        collapse_ops.append(np.sqrt(k_b_to_d) * op(2, 1, dim))
    if k_d_to_b > 0:
        collapse_ops.append(np.sqrt(k_d_to_b) * op(1, 2, dim))
    return tuple(collapse_ops)


def v_system_liouvillian(params):
    return liouvillian_from_hamiltonian_and_lindblad_ops(
        v_system_hamiltonian(params),
        v_system_lindblad_ops(params),
    )


def v_system_mu_components(params):
    dim = 3
    mu_gb = params["mu_gb"]
    mu_gd = params["mu_gd"]
    mu_plus = mu_gb * op(1, 0, dim) + mu_gd * op(2, 0, dim)
    mu_minus = mu_plus.conj().T
    return mu_plus, mu_minus


def v_system_model():
    return general.LiouvilleModel.from_hamiltonian(
        dimension=3,
        hamiltonian=v_system_hamiltonian,
        collapse_ops=v_system_lindblad_ops,
        initial_state=lambda _params: general.vec(projector(0, 3)),
        transition_dipole=lambda params: sum(v_system_mu_components(params)),
        dipole_components=v_system_mu_components,
    )


def v_system_parameter_map(*controls):
    if not controls:
        controls = (general.ControlSpec("J_bd", reference_value=0.0),)
    return general.ParameterMap(
        reference_params={
            "omega_b": 1.0,
            "omega_d": 1.18,
            "J_bd": 0.0,
            "gamma_b": 0.03,
            "gamma_d": 0.004,
            "gamma_phi_b": 0.02,
            "gamma_phi_d": 0.012,
            "k_bd": 0.006,
            "temperature": 0.4,
            "mu_gb": 1.0,
            "mu_gd": 0.08,
        },
        controls=tuple(controls),
    )


def test_v_system_model_construction_is_operator_first_and_qutip_like():
    """Document the intended user-facing construction style.

    A user should be able to define transparent Hilbert-space operators first,
    then assemble H, collapse operators, and mu from those objects. This mirrors
    the QuTiP workflow without requiring QuTiP as a dependency.
    """

    params = v_system_parameter_map().params([0.0])
    dim = 3

    states = {
        "g": ket(0, dim),
        "b": ket(1, dim),
        "d": ket(2, dim),
    }
    projectors = {name: state @ state.conj().T for name, state in states.items()}
    transitions = {
        "b_g": states["b"] @ states["g"].conj().T,
        "d_g": states["d"] @ states["g"].conj().T,
        "g_b": states["g"] @ states["b"].conj().T,
        "g_d": states["g"] @ states["d"].conj().T,
        "b_d": states["b"] @ states["d"].conj().T,
        "d_b": states["d"] @ states["b"].conj().T,
    }
    hamiltonian = (
        params["omega_b"] * projectors["b"]
        + params["omega_d"] * projectors["d"]
        + params["J_bd"] * (transitions["b_d"] + transitions["d_b"])
    )
    mu_plus = params["mu_gb"] * transitions["b_g"] + params["mu_gd"] * transitions["d_g"]
    mu_minus = mu_plus.conj().T
    k_b_to_d, k_d_to_b = detailed_balance_transfer_rates(params)
    c_ops = (
        np.sqrt(params["gamma_b"]) * transitions["g_b"],
        np.sqrt(params["gamma_d"]) * transitions["g_d"],
        np.sqrt(params["gamma_phi_b"]) * projectors["b"],
        np.sqrt(params["gamma_phi_d"]) * projectors["d"],
        np.sqrt(k_b_to_d) * transitions["d_b"],
        np.sqrt(k_d_to_b) * transitions["b_d"],
    )

    np.testing.assert_allclose(
        hamiltonian,
        v_system_hamiltonian(params),
    )
    np.testing.assert_allclose(mu_plus, v_system_mu_components(params)[0])
    np.testing.assert_allclose(mu_minus, v_system_mu_components(params)[1])
    for actual, expected in zip(v_system_lindblad_ops(params), c_ops, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_v_system_liouvillian_matches_density_matrix_lindblad_equation():
    params = v_system_parameter_map().params([0.03])
    hamiltonian = v_system_hamiltonian(params)
    lindblad_ops = v_system_lindblad_ops(params)
    liouvillian = v_system_liouvillian(params)
    rho = np.array(
        [
            [0.6, 0.02 + 0.01j, 0.03 - 0.02j],
            [0.02 - 0.01j, 0.25, 0.04 + 0.03j],
            [0.03 + 0.02j, 0.04 - 0.03j, 0.15],
        ],
        dtype=complex,
    )

    expected = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    for collapse in lindblad_ops:
        cdc = collapse.conj().T @ collapse
        expected += (
            collapse @ rho @ collapse.conj().T
            - 0.5 * (cdc @ rho + rho @ cdc)
        )

    actual = general.mat(liouvillian @ general.vec(rho), dimension=3)

    np.testing.assert_allclose(actual, expected)


def test_qutip_style_adapter_matches_core_v_system_model_without_qutip_dependency():
    params = v_system_parameter_map().params([0.02])
    adapter_model = qutip_model_from_hamiltonian(
        dimension=3,
        hamiltonian=v_system_hamiltonian,
        collapse_ops=v_system_lindblad_ops,
        initial_state=lambda _params: projector(0, 3),
        transition_dipole=lambda p: sum(v_system_mu_components(p)),
        dipole_components=v_system_mu_components,
    )
    core_model = v_system_model()

    np.testing.assert_allclose(adapter_model.L(params), core_model.L(params))
    np.testing.assert_allclose(adapter_model.rho0(params), core_model.rho0(params))
    np.testing.assert_allclose(adapter_model.mu(params), core_model.mu(params))
    for actual, expected in zip(
        adapter_model.mu_components(params),
        core_model.mu_components(params),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


def test_v_system_liouvillian_is_trace_preserving():
    model = v_system_model()
    params = v_system_parameter_map().params([0.05])
    liouvillian = model.L(params)
    trace_row = general.vec(np.eye(3)).conj()

    np.testing.assert_allclose(trace_row @ liouvillian, np.zeros(9), atol=1e-12)


def test_v_system_bright_dark_transfer_rates_obey_detailed_balance():
    params = v_system_parameter_map().params([0.0])
    k_b_to_d, k_d_to_b = detailed_balance_transfer_rates(params)
    delta = params["omega_d"] - params["omega_b"]

    np.testing.assert_allclose(
        k_b_to_d / k_d_to_b,
        np.exp(-delta / params["temperature"]),
    )
    assert k_d_to_b > k_b_to_d


def test_v_system_transfer_rates_allow_explicit_inverse_temperature():
    params = v_system_parameter_map().params([0.0])
    params_with_beta = dict(params)
    params_with_beta.pop("temperature")
    params_with_beta["beta_bd"] = 2.5
    k_b_to_d, k_d_to_b = detailed_balance_transfer_rates(params_with_beta)
    delta = params["omega_d"] - params["omega_b"]

    np.testing.assert_allclose(
        k_b_to_d / k_d_to_b,
        np.exp(-params_with_beta["beta_bd"] * delta),
    )


def test_v_system_zero_temperature_transfer_goes_downhill_only():
    params = v_system_parameter_map().params([0.0])
    params_zero_t = dict(params)
    params_zero_t["temperature"] = 0.0
    k_b_to_d, k_d_to_b = detailed_balance_transfer_rates(params_zero_t)

    assert k_b_to_d == 0.0
    assert k_d_to_b == params["k_bd"]


def test_v_system_dephases_bright_and_dark_states_independently():
    params = v_system_parameter_map().params([0.0])
    base = dict(params)
    no_dephasing = dict(params)
    no_dephasing["gamma_phi_b"] = 0.0
    no_dephasing["gamma_phi_d"] = 0.0

    L_with = v_system_liouvillian(base)
    L_without = v_system_liouvillian(no_dephasing)
    dephasing_piece = L_with - L_without

    assert np.linalg.norm(dephasing_piece) > 0.0
    np.testing.assert_allclose(
        general.vec(np.eye(3)).conj() @ dephasing_piece,
        np.zeros(9),
        atol=1e-12,
    )


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
        general.ControlSpec("J_bd", reference_value=0.0, scale=0.02),
        general.ControlSpec("mu_gd", reference_value=0.08, scale=0.01),
    )

    params = pmap.params([2.0, -3.0])
    assert params["J_bd"] == 0.04
    assert params["mu_gd"] == 0.05
    assert params["omega_b"] == 1.0

    derivatives = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)

    assert set(derivatives) == {(), (0,), (1,), (0, 0), (0, 1), (1, 1)}
    assert derivatives[()].shape == (9, 9)
    assert np.linalg.norm(derivatives[(0,)]) > 0.0
    np.testing.assert_allclose(derivatives[(1,)], np.zeros((9, 9)), atol=1e-10)


def test_v_system_compares_coherent_exchange_and_incoherent_transfer_controls():
    model = v_system_model()
    pmap = v_system_parameter_map(
        general.ControlSpec("J_bd", reference_value=0.0, scale=0.02),
        general.ControlSpec("k_bd", reference_value=0.006, scale=0.003),
    )
    omegas = np.linspace(0.85, 1.3, 6)
    candidates = general.canonical_pathways(("RP",))
    pathways, _candidate_residuals = general.independent_canonical_pathways(
        model,
        pmap,
        omegas,
        candidates,
        stage="after_U2",
    )

    reference = general.liouville_transport_matrix(
        model,
        pmap,
        np.array([0.0, 0.0]),
        omegas,
        pathways,
        stage="after_U2",
    )
    coherent_shift = general.liouville_transport_matrix(
        model,
        pmap,
        np.array([1.0, 0.0]),
        omegas,
        pathways,
        stage="after_U2",
    )
    incoherent_shift = general.liouville_transport_matrix(
        model,
        pmap,
        np.array([0.0, 1.0]),
        omegas,
        pathways,
        stage="after_U2",
    )

    coherent_displacements = pmap.physical_displacements([1.0, 0.0])
    incoherent_displacements = pmap.physical_displacements([0.0, 1.0])
    np.testing.assert_allclose(coherent_displacements["J_bd"], 0.02)
    np.testing.assert_allclose(coherent_displacements["k_bd"], 0.0)
    np.testing.assert_allclose(incoherent_displacements["J_bd"], 0.0)
    np.testing.assert_allclose(incoherent_displacements["k_bd"], 0.003)
    np.testing.assert_allclose(reference.T, np.eye(len(pathways)), atol=1e-9)
    assert np.linalg.norm(coherent_shift.T - reference.T) > 0.0
    assert np.linalg.norm(incoherent_shift.T - reference.T) > 0.0
    assert np.max(coherent_shift.residuals) >= 0.0
    assert np.max(incoherent_shift.residuals) >= 0.0
