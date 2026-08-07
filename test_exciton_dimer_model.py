import numpy as np

import Liouville_transport_general as general
from exciton_dimer_model import (
    detailed_balance_site_transfer_rates,
    excited_state_absorption_frequencies,
    exciton_dimer_hamiltonian,
    exciton_dimer_model,
    exciton_dimer_mu_components,
    exciton_dimer_parameter_map,
)


def test_exciton_dimer_model_is_four_level_and_trace_preserving():
    model = exciton_dimer_model()
    pmap = exciton_dimer_parameter_map()
    params = pmap.params([0.0])
    liouvillian = model.L(params)
    trace_row = general.vec(np.eye(4)).conj()

    assert model.dimension == 4
    assert liouvillian.shape == (16, 16)
    np.testing.assert_allclose(trace_row @ liouvillian, np.zeros(16), atol=1e-12)


def test_exciton_dimer_dipole_includes_ground_and_esa_transitions():
    params = exciton_dimer_parameter_map().params([0.0])
    params["mu_1"] = 1.2
    params["mu_2"] = 0.8
    params["mu_esa_scale"] = 1.0
    mu_plus, mu_minus = exciton_dimer_mu_components(params)

    np.testing.assert_allclose(mu_plus[1, 0], params["mu_1"])
    np.testing.assert_allclose(mu_plus[2, 0], params["mu_2"])
    np.testing.assert_allclose(mu_plus[3, 1], params["mu_2"])
    np.testing.assert_allclose(mu_plus[3, 2], params["mu_1"])
    np.testing.assert_allclose(mu_minus, mu_plus.conj().T)


def test_exciton_dimer_can_disable_esa_transitions():
    params = exciton_dimer_parameter_map().params([0.0])
    params["mu_esa_scale"] = 0.0
    mu_plus, _mu_minus = exciton_dimer_mu_components(params)

    assert np.linalg.norm(mu_plus[1:3, 0]) > 0.0
    np.testing.assert_allclose(mu_plus[3, 1], 0.0)
    np.testing.assert_allclose(mu_plus[3, 2], 0.0)


def test_positive_J_12_brightens_upper_exciton_in_current_convention():
    params = exciton_dimer_parameter_map().params([0.0])
    params["omega_1"] = 1.0
    params["omega_2"] = 1.0
    params["J_12"] = 0.05
    params["mu_1"] = 1.0
    params["mu_2"] = 1.0
    h_one = exciton_dimer_hamiltonian(params)[1:3, 1:3]
    energies, vectors = np.linalg.eigh(h_one)
    site_dipole = np.array([params["mu_1"], params["mu_2"]], dtype=complex)
    oscillator_strengths = np.abs(site_dipole.conj() @ vectors) ** 2

    np.testing.assert_allclose(energies, [0.95, 1.05])
    assert oscillator_strengths[1] > oscillator_strengths[0]


def test_exciton_dimer_reports_esa_transition_frequencies():
    params = exciton_dimer_parameter_map().params([0.0])
    params["J_12"] = 0.0
    params["anharmonicity"] = -0.06

    frequencies = excited_state_absorption_frequencies(params)

    np.testing.assert_allclose(frequencies, [1.12, 0.94])


def test_exciton_dimer_site_transfer_rates_obey_detailed_balance():
    params = exciton_dimer_parameter_map().params([0.0])
    k_1_to_2, k_2_to_1 = detailed_balance_site_transfer_rates(params)
    delta = params["omega_2"] - params["omega_1"]

    np.testing.assert_allclose(
        k_1_to_2 / k_2_to_1,
        np.exp(-delta / params["temperature"]),
    )
