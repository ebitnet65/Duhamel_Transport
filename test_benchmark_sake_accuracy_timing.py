import json
from types import SimpleNamespace

import numpy as np

import benchmark_sake_accuracy_timing as benchmark
import fit_sake_dimer_iterative as iterative


def test_aggregate_relative_error_uses_all_blocks():
    actual = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
    predicted = [np.array([0.0, 0.0]), np.array([0.0, 1.0])]
    expected = np.sqrt(2.0) / np.sqrt(5.0)
    assert np.isclose(benchmark.aggregate_relative_error(actual, predicted), expected)


def test_load_derivative_cache_round_trip(tmp_path):
    cache = tmp_path / "derivatives.npz"
    root = np.eye(2, dtype=complex)
    first = np.ones((2, 2), dtype=complex)
    np.savez_compressed(cache, root=root, d_0=first)
    cache.with_suffix(".json").write_text(
        json.dumps({"derivative_seconds": 1.25}),
        encoding="utf-8",
    )

    derivatives, seconds, metadata = benchmark.load_derivative_cache(cache)

    np.testing.assert_allclose(derivatives[()], root)
    np.testing.assert_allclose(derivatives[(0,)], first)
    assert seconds == 1.25
    assert metadata["derivative_seconds"] == 1.25


def test_normalized_direction():
    args = SimpleNamespace(direction_J=-1.0, direction_k=1.0)
    direction = benchmark.normalized_direction(args)
    np.testing.assert_allclose(direction, [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_clean_dimer_uses_omega0_scaled_controls_and_asymmetric_sites():
    args = SimpleNamespace(
        omega0=2.0,
        delta_omega=0.18,
        delta_J=None,
        delta_k=None,
        reference_J=0.0,
        reference_k=0.0,
        clean_gamma=0.01,
        clean_gamma_phi=0.01,
        clean_mu_esa_scale=1.0,
    )
    parameter_map = benchmark.clean_dimer_parameter_map(args)

    assert np.isclose(parameter_map.reference_params["omega_1"], 2.09)
    assert np.isclose(parameter_map.reference_params["omega_2"], 1.91)
    assert np.isclose(parameter_map.controls[0].scale, 2.0)
    assert np.isclose(parameter_map.controls[1].scale, 2.0)


def test_derivative_cache_validation_rejects_wrong_reference():
    args = SimpleNamespace(
        omega0=2.0,
        delta_omega=0.18,
        delta_J=None,
        delta_k=None,
        reference_J=0.0,
        reference_k=0.0,
        clean_gamma=0.01,
        clean_gamma_phi=0.01,
        clean_mu_esa_scale=1.0,
    )
    parameter_map = benchmark.clean_dimer_parameter_map(args)
    metadata = {
        "omega0": 1.0,
        "delta_omega": 0.18,
        "reference_J": 0.0,
        "reference_k": 0.0,
        "scale_J": 2.0,
        "scale_k": 2.0,
    }

    with np.testing.assert_raises_regex(ValueError, "does not match"):
        benchmark.validate_derivative_cache(metadata, parameter_map, args)


def test_iterative_normalized_squared_error_aggregates_blocks():
    actual = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
    predicted = [np.array([0.0, 0.0]), np.array([0.0, 1.0])]
    assert np.isclose(iterative.normalized_squared_error(actual, predicted), 2.0 / 5.0)


def test_exact_step_transport_orientation():
    reference = [np.eye(2, dtype=complex)]
    first = np.array([[1.0, 0.2], [0.1, 1.0]], dtype=complex)
    second = np.array([[0.9, -0.1], [0.3, 1.1]], dtype=complex)
    state_one = [reference[0] @ first.T]
    state_two = [state_one[0] @ second.T]

    step_one = iterative.exact_step_transport(reference, state_one, 1e-12)
    step_two = iterative.exact_step_transport(state_one, state_two, 1e-12)

    np.testing.assert_allclose(step_two @ step_one, second @ first)
