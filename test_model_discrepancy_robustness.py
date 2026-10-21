import json

import numpy as np

from model_discrepancy_robustness import (
    ReliabilityThresholds,
    analyze_model_discrepancy_sweep,
    decompose_residual,
)


def test_residual_decomposition_separates_tangent_and_model_error():
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    residual = np.array([2.0, -1.0, 3.0])
    parallel, orthogonal, singular_values, rank = decompose_residual(residual, jacobian)
    np.testing.assert_allclose(parallel, [2.0, -1.0, 0.0])
    np.testing.assert_allclose(orthogonal, [0.0, 0.0, 3.0])
    assert np.vdot(parallel, orthogonal) == 0.0
    assert rank == 2
    np.testing.assert_allclose(singular_values, [1.0, 1.0])


def test_noise_free_artifact_sweep_finds_explicit_reliability_window():
    truth = np.array([0.4, -0.2])

    def ideal(q):
        return np.array([q[0], q[1], q[0] + q[1]])

    # Artifact has an exactly tangent component (absorbed as endpoint bias) and
    # a normal component (irreducible structured residual).
    normal = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    tangent_shift = np.array([0.5, 0.0, 0.5])

    def distorted(strength, seed):
        assert seed is None
        return ideal(truth) + strength * (tangent_shift + normal)

    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def recover(observed):
        endpoint, *_ = np.linalg.lstsq(jacobian, observed, rcond=None)
        return {"endpoint": endpoint, "success": True}

    thresholds = ReliabilityThresholds(
        maximum_relative_residual=0.2,
        maximum_relative_orthogonal_residual=0.2,
        maximum_orthogonal_residual_fraction=1.0,
        maximum_euclidean_endpoint_bias=0.09,
        maximum_physical_endpoint_bias=0.09,
        maximum_jacobian_condition_number=10.0,
    )
    result = analyze_model_discrepancy_sweep(
        truth,
        [0.0, 0.05, 0.10, 0.20],
        distorted,
        ideal,
        recover,
        lambda _q: jacobian,
        lambda _q: np.eye(2),
        thresholds=thresholds,
        downstream_fn=lambda q: {"route_endpoint": q, "holonomy": np.outer(q, q)},
    )
    assert result.uncertainty_window == (0.0, 0.1)
    assert [item.reliable for item in result.diagnostics] == [True, True, True, False]
    assert "endpoint_bias" in result.diagnostics[-1].failure_reasons
    assert result.diagnostics[-1].tangent_orthogonal_residual_norm > 0.0
    assert result.diagnostics[-1].physical_uncertainty_rms > 0.0
    assert result.diagnostics[-1].downstream_changes["holonomy"] > 0.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_replicate_api_supports_future_noise_ensembles():
    truth = np.array([0.2])
    result = analyze_model_discrepancy_sweep(
        truth,
        [0.0, 0.1],
        lambda strength, seed: np.array([truth[0] + strength + 1e-4 * seed]),
        lambda q: q,
        lambda observed: observed,
        lambda _q: np.ones((1, 1)),
        lambda _q: np.eye(1),
        replicate_seeds=(1, 2, 3),
        thresholds=ReliabilityThresholds(
            maximum_relative_residual=1.0,
            maximum_relative_orthogonal_residual=1.0,
            maximum_orthogonal_residual_fraction=1.0,
            maximum_euclidean_endpoint_bias=1.0,
            maximum_physical_endpoint_bias=1.0,
        ),
    )
    assert len(result.diagnostics) == 6
    assert result.replicate_seeds == (1, 2, 3)
