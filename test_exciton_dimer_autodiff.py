import numpy as np

import Liouville_autodiff as autodiff
import Liouville_transport_general as general
from exciton_dimer_autodiff import jax_exciton_dimer_liouvillian
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map


def test_jax_dimer_liouvillian_matches_numpy_reference_model():
    model = exciton_dimer_model()
    pmap = exciton_dimer_parameter_map()
    params = pmap.reference_params

    actual = np.asarray(jax_exciton_dimer_liouvillian(params), dtype=complex)
    expected = model.L(params)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jax_dimer_liouvillian_derivatives_have_expected_linear_controls():
    pmap = general.ParameterMap(
        reference_params={
            **exciton_dimer_parameter_map().reference_params,
            "J_12": 0.0,
            "k_12": 0.0,
            "anharmonicity": 0.0,
            "gamma_1": 0.01,
            "gamma_2": 0.01,
            "gamma_phi_1": 0.01,
            "gamma_phi_2": 0.01,
            "mu_1": 1.0,
            "mu_2": 1.0,
            "mu_esa_scale": 1.0,
        },
        controls=(
            general.ControlSpec("J_12", reference_value=0.0, scale=0.045),
            general.ControlSpec("k_12", reference_value=0.0, scale=0.012),
        ),
    )

    derivatives = autodiff.jax_liouvillian_derivatives(
        jax_exciton_dimer_liouvillian,
        pmap,
        liouville_dimension=16,
    )

    assert set(derivatives) == {(), (0,), (1,), (0, 0), (0, 1), (1, 1)}
    assert np.linalg.norm(derivatives[(0,)]) > 0.0
    assert np.linalg.norm(derivatives[(1,)]) > 0.0
    np.testing.assert_allclose(derivatives[(0, 0)], np.zeros((16, 16)), atol=1e-13)
    np.testing.assert_allclose(derivatives[(0, 1)], np.zeros((16, 16)), atol=1e-13)
    np.testing.assert_allclose(derivatives[(1, 1)], np.zeros((16, 16)), atol=1e-13)
