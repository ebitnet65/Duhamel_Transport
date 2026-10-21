import numpy as np
import pytest

import Liouville_covariant_transport as lct
import Liouville_geodesics as lg
import optical_bloch_geometry as obg
from dimer_covariant_geometry import build_dimer_geometry_backend


@pytest.fixture(scope="module")
def calculation():
    if not obg.is_jax_available():
        pytest.skip("JAX is required for the three-control dimer test.")
    geometry = build_dimer_geometry_backend(
        control_mode="coupling_transfer_dephasing",
        fixed_detuning=0.18,
        beta_12=2.5,
    )
    start = np.asarray([0.070, 0.003, 0.004])
    truth = np.asarray([-0.025, 0.020, 0.028])
    straight = lg.linear_path(start, truth, 9)
    geodesic = lg.solve_discrete_geodesic(
        start,
        truth,
        geometry.metric,
        n_vertices=9,
        bounds=((-0.10, 0.10), (1e-4, 0.03), (1e-4, 0.04)),
    )
    frame0 = lct.metric_orthonormal_frame(geometry.metric(start))
    straight_transport = lct.parallel_transport_frame(
        straight, frame0, geometry.christoffel, metric_fn=geometry.metric
    )
    geodesic_transport = lct.parallel_transport_frame(
        geodesic.path, frame0, geometry.christoffel, metric_fn=geometry.metric
    )
    return geometry, straight, geodesic, straight_transport, geodesic_transport


def test_three_control_geodesic_reaches_truth_and_lowers_energy(calculation):
    _, straight, geodesic, _, _ = calculation
    assert geodesic.success, geodesic.message
    np.testing.assert_array_equal(geodesic.path[0], straight[0])
    np.testing.assert_array_equal(geodesic.path[-1], straight[-1])
    assert geodesic.energy < geodesic.initial_energy
    assert np.all(geodesic.metric_determinants > 0.0)
    assert np.max(geodesic.metric_condition_numbers) < 1e4


def test_metric_encodes_detailed_balance_and_collective_dephasing(calculation):
    geometry, straight, _, _, _ = calculation
    coupling, transfer_rate, dephasing_rate = straight[0]
    del coupling
    beta = 2.5
    delta = 0.18
    expected = np.diag(
        [
            2.0,
            (np.exp(-0.5 * beta * delta) + np.exp(0.5 * beta * delta))
            / (4.0 * transfer_rate),
            2.0 / dephasing_rate,
        ]
    )
    np.testing.assert_allclose(geometry.metric(straight[0]), expected, rtol=1e-12)


def test_rate_coordinates_depart_from_straight_interpolation(calculation):
    _, straight, geodesic, _, _ = calculation
    displacement = np.max(np.abs(geodesic.path - straight), axis=0)
    scales = np.ptp(straight, axis=0)
    assert displacement[1] / scales[1] > 1e-3
    assert displacement[2] / scales[2] > 1e-3
    np.testing.assert_allclose(geodesic.path[:, 0], straight[:, 0], atol=2e-6)


def test_three_control_paths_give_metric_preserving_transport(calculation):
    _, _, _, straight_transport, geodesic_transport = calculation
    assert straight_transport.success and geodesic_transport.success
    assert np.max(straight_transport.metric_gram_error) < 2e-6
    assert np.max(geodesic_transport.metric_gram_error) < 2e-6
    # This square-root generator embedding is intrinsically flat, so parallel
    # transport is path independent even though its rate-coordinate geodesics
    # are curved in the original controls.
    np.testing.assert_allclose(
        geodesic_transport.final_frame,
        straight_transport.final_frame,
        rtol=2e-7,
        atol=2e-7,
    )
