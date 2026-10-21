import json

import numpy as np
import pytest

import optical_bloch_geometry as obg
from benchmark_recovered_endpoint_transport import (
    _endpoint_from_json,
    compare_routes_to_endpoint,
    write_results,
)


@pytest.fixture(scope="module")
def route_comparison():
    if not obg.is_jax_available():
        pytest.skip("JAX is required for the recovered-endpoint route test.")
    reference = np.asarray([0.070, 0.003, 0.004])
    q_star = np.asarray([-0.025, 0.020, 0.028])
    diagnostics, arrays = compare_routes_to_endpoint(
        reference, q_star, n_vertices=9
    )
    return diagnostics, arrays


def test_route_reaches_fixed_recovered_endpoint(route_comparison):
    diagnostics, arrays = route_comparison
    np.testing.assert_array_equal(arrays["geodesic_path"][0], arrays["reference"])
    np.testing.assert_array_equal(arrays["geodesic_path"][-1], arrays["q_star"])
    assert diagnostics["geodesic"]["endpoint_error"] == 0.0
    assert diagnostics["geodesic"]["energy"] < diagnostics["geodesic"]["straight_energy"]
    assert diagnostics["geodesic"]["relative_energy_reduction"] > 0.01


def test_both_routes_preserve_the_metric_frame(route_comparison):
    diagnostics, _ = route_comparison
    assert diagnostics["transport"]["straight_max_gram_error"] < 2.0e-6
    assert diagnostics["transport"]["geodesic_max_gram_error"] < 2.0e-6
    # This collapse-operator embedding is flat.  The curved coordinate path
    # therefore gives endpoint transport consistent with the straight route.
    assert diagnostics["transport"]["endpoint_frame_frobenius_difference"] < 2.0e-6


def test_results_are_compact_and_machine_readable(route_comparison, tmp_path):
    diagnostics, arrays = route_comparison
    write_results(diagnostics, arrays, tmp_path)

    recovered = json.loads(
        (tmp_path / "transport_diagnostics.json").read_text(encoding="utf-8")
    )
    saved = np.load(tmp_path / "transport_paths.npz")
    assert recovered["q_star"] == diagnostics["q_star"]
    assert saved["geodesic_path"].shape == (9, 3)
    assert set(saved.files) >= {"straight_path", "geodesic_path", "straight_frames"}


def test_endpoint_loader_accepts_upstream_recovery_schema(tmp_path):
    recovery_file = tmp_path / "recovery.json"
    recovery_file.write_text(
        json.dumps({"recovery": {"endpoint": [0.025, 0.014, 0.022]}}),
        encoding="utf-8",
    )
    np.testing.assert_array_equal(
        _endpoint_from_json(recovery_file), [0.025, 0.014, 0.022]
    )
