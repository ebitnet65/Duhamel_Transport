import numpy as np
import pytest


pytest.importorskip("jax")

from dimer_endpoint_inversion import (  # noqa: E402
    EndpointRecoveryConfig,
    generate_synthetic_truth,
    recover_dimer_endpoint,
)
from dimer_spectral_observable import (  # noqa: E402
    DimerSpectrumConfig,
    build_packed_dimer_spectral_observable,
)


@pytest.fixture(scope="module")
def recovery():
    frequencies = np.linspace(0.86, 1.32, 5)
    observable = build_packed_dimer_spectral_observable(
        DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    )
    truth_endpoint = np.asarray([0.025, 0.014, 0.022])
    truth = generate_synthetic_truth(observable, truth_endpoint, 0.8, seed=17)
    config = EndpointRecoveryConfig(
        starts=((0.04, 0.006, 0.012), (-0.04, 0.025, 0.035)),
        n_random_starts=0,
        seed=17,
    )
    result = recover_dimer_endpoint(
        observable,
        truth.observation,
        truth.t2,
        config=config,
        truth_endpoint=truth_endpoint,
    )
    return truth, result


def test_noiseless_multistart_recovers_fixed_endpoint(recovery):
    truth, result = recovery
    assert result.success, result.message
    np.testing.assert_allclose(result.endpoint, truth.endpoint, rtol=2e-6, atol=2e-8)
    assert result.endpoint_absolute_error < 1e-7
    assert result.endpoint_relative_error < 3e-6
    assert result.relative_spectral_residual < 1e-8
    assert result.objective < 1e-16


def test_recovery_reports_identifiability_and_all_runs(recovery):
    _, result = recovery
    assert result.jacobian_rank == 3
    assert result.jacobian_singular_values.shape == (3,)
    assert np.all(result.jacobian_singular_values > 0.0)
    assert np.isfinite(result.jacobian_condition_number)
    assert len(result.runs) == 3  # midpoint plus two supplied starts
    for run in result.runs:
        assert np.isfinite(run.objective)
        assert run.endpoint.shape == (3,)


def test_synthetic_noise_is_seeded_and_scaled():
    frequencies = np.linspace(0.9, 1.28, 3)
    observable = build_packed_dimer_spectral_observable(
        DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    )
    q = np.asarray([0.03, 0.01, 0.02])
    first = generate_synthetic_truth(observable, q, 0.5, relative_noise=0.02, seed=8)
    second = generate_synthetic_truth(observable, q, 0.5, relative_noise=0.02, seed=8)
    np.testing.assert_array_equal(first.observation, second.observation)
    assert not np.array_equal(first.observation, first.noiseless_observation)


def test_results_are_json_serializable(recovery):
    import json

    truth, result = recovery
    json.dumps({"truth": truth.to_dict(), "recovery": result.to_dict()}, allow_nan=False)
