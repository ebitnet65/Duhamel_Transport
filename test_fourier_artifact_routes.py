import json
import numpy as np
from benchmark_fourier_artifact_routes import run_artifact_route_sweep


def test_small_artifact_route_sweep_is_auditable_and_path_resolved():
    payload, arrays = run_artifact_route_sweep(
        [0.0, 0.005], n_omega=3, vertices=4, refinement_factors=(1, 2)
    )
    diagnostics = payload["discrepancy"]["diagnostics"]
    assert len(diagnostics) == 2 and len(payload["routes"]) == 2
    assert diagnostics[0]["endpoint_bias"] < 1e-6
    for index in range(2):
        np.testing.assert_array_equal(arrays[f"geodesic_path_{index}"][-1], arrays[f"endpoint_{index}"])
        assert np.all(np.isfinite(arrays[f"covariance_{index}"]))
    json.dumps(payload, allow_nan=False)
