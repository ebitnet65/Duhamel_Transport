import numpy as np
import pytest

import Liouville_covariant_transport as lct
import Liouville_geodesics as lg
import optical_bloch_geometry as obg
from dimer_covariant_geometry import build_dimer_geometry_backend


@pytest.fixture(scope="module")
def comparison():
    if not obg.is_jax_available():
        pytest.skip("JAX is required for the dimer geodesic test.")
    geometry = build_dimer_geometry_backend(beta_12=10.0)
    start = np.asarray([0.24, 0.075, 0.003])
    stop = np.asarray([0.03, 0.005, 0.020])
    straight = lg.linear_path(start, stop, 8)
    geodesic = lg.solve_discrete_geodesic(
        start,
        stop,
        geometry.metric,
        n_vertices=8,
        bounds=((0.01, 0.28), (-0.10, 0.10), (1e-4, 0.03)),
    )
    frame0 = lct.metric_orthonormal_frame(geometry.metric(start))
    straight_transport = lct.parallel_transport_frame(
        straight, frame0, geometry.christoffel, metric_fn=geometry.metric
    )
    geodesic_transport = lct.parallel_transport_frame(
        geodesic.path, frame0, geometry.christoffel, metric_fn=geometry.metric
    )
    return geometry, straight, geodesic, straight_transport, geodesic_transport


def test_dimer_geodesic_has_fixed_endpoints_and_lower_energy(comparison):
    _, straight, geodesic, _, _ = comparison
    assert geodesic.success, geodesic.message
    np.testing.assert_array_equal(geodesic.path[0], straight[0])
    np.testing.assert_array_equal(geodesic.path[-1], straight[-1])
    assert geodesic.energy < geodesic.initial_energy
    assert geodesic.relative_energy_reduction > 1e-3
    assert np.all(geodesic.metric_determinants > 0.0)
    assert np.max(geodesic.metric_condition_numbers) < 12.0


def test_geodesic_and_straight_paths_produce_distinct_valid_transport(comparison):
    _, _, _, straight_transport, geodesic_transport = comparison
    assert straight_transport.success and geodesic_transport.success
    assert np.max(straight_transport.metric_gram_error) < 2e-7
    assert np.max(geodesic_transport.metric_gram_error) < 2e-7
    frame_difference = np.linalg.norm(
        geodesic_transport.final_frame - straight_transport.final_frame, ord="fro"
    )
    assert frame_difference > 1e-3
