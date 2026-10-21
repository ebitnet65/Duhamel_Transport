import json

import numpy as np

from benchmark_dimer_pathway_parallel_transport import run_benchmark


def test_fixed_endpoint_pathway_transport_is_ordered_stable_and_convergent():
    diagnostics, arrays = run_benchmark(
        [0.070, 0.003, 0.004], [0.025, 0.014, 0.022],
        n_vertices=5, n_omega=3, refinement_factors=(1, 2),
    )
    assert diagnostics["n_pathways"] > 0
    assert diagnostics["pathway_sectors"] == sorted(
        diagnostics["pathway_sectors"], key=lambda value: {"RP": 0, "NRP": 1}[value]
    )
    assert diagnostics["physical_route"]["endpoint_error"] == 0.0
    assert diagnostics["route_comparison"]["endpoint_spectral_difference"] == 0.0
    assert diagnostics["straight"]["maximum_cumulative_gram_error"] < 1e-10
    assert diagnostics["geodesic"]["maximum_cumulative_gram_error"] < 1e-10
    assert diagnostics["straight"]["successive_transport_errors"][0] >= 0.0
    assert diagnostics["geodesic"]["successive_transport_errors"][0] >= 0.0
    assert np.all(np.isfinite(arrays["straight_endpoint_spectrum"]))
    json.dumps(diagnostics, allow_nan=False)
