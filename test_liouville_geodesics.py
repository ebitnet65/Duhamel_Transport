import numpy as np
import pytest

import Liouville_geodesics as lg


def test_euclidean_geodesic_is_the_straight_line():
    metric = lambda lam: np.eye(len(lam))
    start = np.asarray([-0.2, 0.1, 0.4])
    stop = np.asarray([0.5, -0.3, 0.2])
    expected = lg.linear_path(start, stop, 7)

    result = lg.solve_discrete_geodesic(start, stop, metric, n_vertices=7)

    assert result.success, result.message
    np.testing.assert_allclose(result.path, expected, atol=1e-9)
    np.testing.assert_allclose(
        result.energy, 0.5 * np.dot(stop - start, stop - start), atol=1e-14
    )


def test_path_energy_is_invariant_under_uniform_refinement_for_constant_metric():
    metric_matrix = np.diag([2.0, 3.0])
    metric = lambda _lam: metric_matrix
    start = np.asarray([0.1, -0.2])
    stop = np.asarray([0.7, 0.5])

    coarse = lg.discrete_path_energy(lg.linear_path(start, stop, 3), metric)
    fine = lg.discrete_path_energy(lg.linear_path(start, stop, 21), metric)

    np.testing.assert_allclose(coarse, fine, atol=1e-14)


def test_geodesic_rejects_endpoint_outside_bounds():
    with pytest.raises(ValueError, match=r"start\[0\].*lower bound"):
        lg.solve_discrete_geodesic(
            np.asarray([-0.1, 0.2]),
            np.asarray([0.5, 0.6]),
            lambda _lam: np.eye(2),
            bounds=((0.0, 1.0), (0.0, 1.0)),
        )
