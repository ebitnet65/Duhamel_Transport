import numpy as np

import Liouville_transport_general as general
import phase_matched_response as reference


PHYSICAL_LABELS = {
    ("RP", "L-L+L+"): "RP-GSB",
    ("RP", "L-R+L+"): "RP-SE",
    ("NRP", "L+L-L+"): "NRP-GSB",
    ("NRP", "L+R-L+"): "NRP-SE",
}


def two_level_model():
    return general.LiouvilleModel(
        dimension=2,
        liouvillian=lambda params: reference.liouvillian(
            w0=params["w0"],
            gamma_phi=params["gamma_phi"],
            theta=params["theta"],
            gamma1=params["gamma1"],
        ),
        initial_state=lambda params: reference.initial_state(
            reference.liouvillian(
                w0=params["w0"],
                gamma_phi=params["gamma_phi"],
                theta=params["theta"],
                gamma1=params["gamma1"],
            ),
            mode="fixed_steady0",
            L_ref=reference.liouvillian(
                w0=params["w0"],
                gamma_phi=params["gamma_phi"],
                theta=0.0,
                gamma1=params["gamma1"],
            ),
        ),
        transition_dipole=lambda _params: reference.mu,
    )


def parameter_map():
    return general.ParameterMap(
        reference_params={
            "w0": 1.0,
            "gamma_phi": 0.1,
            "gamma1": 0.02,
            "theta": 0.0,
        },
        controls=(general.ControlSpec("theta", reference_value=0.0),),
    )


def test_general_pathway_bookkeeping_matches_reference():
    assert general.pathway_label((("L", "-"), ("L", "+"), ("L", "+"))) == "L-L+L+"

    for sector in ("RP", "NRP", "DQC"):
        general_labels = sorted(general.pathway_label(p) for p in general.get_sector_pathways(sector))
        reference_labels = sorted(reference.pathway_label(p) for p in reference.get_sector_pathways(sector))
        assert general_labels == reference_labels


def test_general_pathway_state_matches_reference_implementation():
    model = two_level_model()
    pmap = parameter_map()
    theta = np.array([0.07])
    omega = 1.0
    pathway = (("L", "-"), ("L", "+"), ("L", "+"))

    actual = general.pathway_state_point(
        model,
        pmap,
        theta,
        omega,
        omega,
        pathway,
        "RP",
        stage="after_G3",
    )

    L = reference.liouvillian(theta=theta[0])
    L_ref = reference.liouvillian(theta=0.0)
    rho0 = reference.initial_state(L, mode="fixed_steady0", L_ref=L_ref)
    J1 = reference.interaction_super(*pathway[0])
    J2 = reference.interaction_super(*pathway[1])
    J3 = reference.interaction_super(*pathway[2])
    G1 = reference.G_freq(L, -omega, eta=1e-3)
    G3 = reference.G_freq(L, omega, eta=1e-3)
    U2 = __import__("scipy.linalg").linalg.expm(L * 1.0)
    expected = G3 @ J3 @ U2 @ J2 @ G1 @ J1 @ rho0

    np.testing.assert_allclose(actual, expected)


def test_general_transport_matrix_reduces_to_identity_at_zero_control():
    model = two_level_model()
    pmap = parameter_map()
    omegas = np.linspace(0.8, 1.2, 5)
    pathways = [
        item
        for item in general.canonical_pathways(("RP",), label_map=PHYSICAL_LABELS)
        if item.label in ("RP-GSB", "RP-SE")
    ]

    result = general.liouville_transport_matrix(
        model,
        pmap,
        np.array([0.0]),
        omegas,
        pathways,
        stage="after_U2",
    )

    np.testing.assert_allclose(result.T, np.eye(len(pathways)), atol=1e-10)
    assert result.rank == len(pathways)
    assert result.labels == tuple(item.label for item in pathways)
    assert np.max(result.residuals) < 1e-10


def test_control_map_uses_reference_model_and_uncertainty_scales():
    pmap = general.ParameterMap(
        reference_params={"theta": 0.2, "gamma_phi": 0.1, "w0": 1.0},
        controls=(
            general.ControlSpec("theta", reference_value=0.2, scale=0.05),
            general.ControlSpec("gamma_phi", reference_value=0.1, scale=0.01),
        ),
    )

    reference_params = pmap.params(np.array([0.0, 0.0]))
    shifted_params = pmap.params(np.array([2.0, 3.0]))
    displacements = pmap.physical_displacements(np.array([2.0, 3.0]))

    assert reference_params["theta"] == 0.2
    assert reference_params["gamma_phi"] == 0.1
    np.testing.assert_allclose(shifted_params["theta"], 0.3)
    np.testing.assert_allclose(shifted_params["gamma_phi"], 0.13)
    assert shifted_params["w0"] == 1.0
    np.testing.assert_allclose(displacements["theta"], 0.1)
    np.testing.assert_allclose(displacements["gamma_phi"], 0.03)


def test_finite_difference_liouvillian_derivative_has_control_indices():
    model = two_level_model()
    pmap = parameter_map()

    derivatives = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)

    assert set(derivatives) == {(), (0,), (0, 0)}
    assert derivatives[()].shape == (4, 4)
    assert np.linalg.norm(derivatives[(0,)]) > 0.0
