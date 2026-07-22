import numpy as np

import Liouville_transport_general as general
from fit_v_system_transport_signal import (
    duhamel_transport_expansion_2d,
    finite_difference_transport_expansion_2d,
)
from Liouville_transport_expansion import (
    _stacked_pathway_derivative_matrices,
    duhamel_pathway_derivative_matrices,
    duhamel_transport_expansion,
    finite_difference_transport_expansion,
)
from test_v_system_general import v_system_model, v_system_parameter_map


def _v_system_basis():
    model = v_system_model()
    pmap = v_system_parameter_map(
        general.ControlSpec("J_bd", reference_value=0.0, scale=0.02),
        general.ControlSpec("k_bd", reference_value=0.006, scale=0.003),
    )
    omegas = np.linspace(0.85, 1.3, 6)
    candidates = general.canonical_pathways(("RP",))
    pathways, _ = general.independent_canonical_pathways(
        model,
        pmap,
        omegas,
        candidates,
        stage="after_U2",
    )
    return model, pmap, omegas, pathways


def test_transport_expansion_has_expected_shapes_and_identity_reference():
    model, pmap, omegas, pathways = _v_system_basis()
    l_derivs = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)

    expansion = finite_difference_transport_expansion(
        model,
        pmap,
        omegas,
        pathways,
        liouvillian_derivatives=l_derivs,
        stage="after_U2",
        step=1e-4,
    )

    n = len(pathways)
    np.testing.assert_allclose(expansion.T0, np.eye(n), atol=1e-9)
    assert set(expansion.first) == {0, 1}
    assert set(expansion.second) == {(0, 0), (0, 1), (1, 1)}
    for matrix in expansion.first.values():
        assert matrix.shape == (n, n)
    for matrix in expansion.second.values():
        assert matrix.shape == (n, n)


def test_transport_expansion_reconstructs_exact_projected_T_for_small_controls():
    model, pmap, omegas, pathways = _v_system_basis()
    expansion = finite_difference_transport_expansion(
        model,
        pmap,
        omegas,
        pathways,
        stage="after_U2",
        step=5e-5,
    )
    lam = np.array([0.2, -0.15])
    exact = general.liouville_transport_matrix(
        model,
        pmap,
        lam,
        omegas,
        pathways,
        stage="after_U2",
    )
    approx1 = expansion.evaluate(lam, order=1)
    approx2 = expansion.evaluate(lam, order=2)

    assert np.linalg.norm(approx2 - exact.T) < np.linalg.norm(approx1 - exact.T)
    assert np.linalg.norm(approx2 - exact.T) < 2e-5


def test_transport_expansion_exposes_coherent_and_incoherent_mixing_terms():
    model, pmap, omegas, pathways = _v_system_basis()
    expansion = finite_difference_transport_expansion(
        model,
        pmap,
        omegas,
        pathways,
        stage="after_U2",
    )

    np.testing.assert_allclose(expansion.first[0], np.zeros_like(expansion.first[0]))
    assert np.linalg.norm(expansion.second[(0, 0)]) > 0.0
    assert np.linalg.norm(expansion.first[1]) > 0.0
    assert np.max(expansion.first_residuals[1]) < 1.0


def test_duhamel_pathway_derivatives_match_finite_difference_backend():
    model, pmap, omegas, pathways = _v_system_basis()
    l_derivs = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)

    P0_duh, first_duh, second_duh = duhamel_pathway_derivative_matrices(
        model,
        pmap,
        omegas[:3],
        pathways,
        liouvillian_derivatives=l_derivs,
        stage="after_U2",
        n_quad=32,
    )
    P0_fd, first_fd, second_fd = _stacked_pathway_derivative_matrices(
        model,
        pmap,
        omegas[:3],
        pathways,
        stage="after_U2",
        tau2=1.0,
        eta=1e-3,
        step=1e-3,
    )

    np.testing.assert_allclose(P0_duh, P0_fd, rtol=1e-11, atol=1e-11)
    for key in first_duh:
        np.testing.assert_allclose(first_duh[key], first_fd[key], rtol=2e-5, atol=2e-6)
    for key in second_duh:
        np.testing.assert_allclose(second_duh[key], second_fd[key], rtol=2e-4, atol=2e-5)


def test_duhamel_transport_expansion_matches_finite_difference_transport():
    model, pmap, omegas, pathways = _v_system_basis()
    l_derivs = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)
    fd = finite_difference_transport_expansion(
        model,
        pmap,
        omegas[:3],
        pathways,
        stage="after_U2",
        step=1e-3,
    )
    duh = duhamel_transport_expansion(
        model,
        pmap,
        omegas[:3],
        pathways,
        liouvillian_derivatives=l_derivs,
        stage="after_U2",
        n_quad=32,
    )

    np.testing.assert_allclose(duh.T0, fd.T0, rtol=1e-11, atol=1e-11)
    for key in duh.first:
        np.testing.assert_allclose(duh.first[key], fd.first[key], rtol=5e-5, atol=5e-6)
    for key in duh.second:
        np.testing.assert_allclose(duh.second[key], fd.second[key], rtol=5e-4, atol=5e-5)


def test_duhamel_2d_transport_expansion_matches_finite_difference_transport():
    model, pmap, _omegas, pathways = _v_system_basis()
    omega1 = np.linspace(0.9, 1.15, 2)
    omega3 = np.linspace(0.92, 1.17, 2)
    l_derivs = general.finite_difference_liouvillian_derivatives(model, pmap, step=1e-4)
    fd = finite_difference_transport_expansion_2d(
        model,
        pmap,
        omega1,
        omega3,
        pathways,
        tau2=1.0,
        eta=1e-3,
        step=1e-3,
        second_step=1e-2,
        rcond=1e-12,
    )
    duh = duhamel_transport_expansion_2d(
        model,
        pmap,
        omega1,
        omega3,
        pathways,
        liouvillian_derivatives=l_derivs,
        tau2=1.0,
        eta=1e-3,
        n_quad=32,
        rcond=1e-12,
    )

    np.testing.assert_allclose(duh.T0, fd.T0, rtol=1e-10, atol=1e-10)
    for key in duh.first:
        np.testing.assert_allclose(duh.first[key], fd.first[key], rtol=8e-5, atol=8e-6)
    for key in duh.second:
        np.testing.assert_allclose(duh.second[key], fd.second[key], rtol=8e-3, atol=8e-4)
