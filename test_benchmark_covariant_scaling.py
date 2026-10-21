import numpy as np

from benchmark_covariant_scaling import benchmark_dimension


def test_covariant_scaling_benchmark_smoke():
    records = [benchmark_dimension(dim, n_vertices=4, repeats=2) for dim in (2, 3)]
    for record in records:
        assert record.compile_seconds > 0.0
        assert record.metric_eval_ms > 0.0
        assert record.christoffel_eval_ms > 0.0
        assert record.geodesic_seconds > 0.0
        assert record.transport_seconds > 0.0
        assert record.geodesic_energy <= record.straight_energy
        assert record.max_metric_condition < 100.0
        assert record.max_gram_error < 1e-6
        assert np.isfinite(record.relative_energy_reduction)
