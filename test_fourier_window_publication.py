from pathlib import Path

import numpy as np

from fourier_window_publication import load_checked_benchmark
from generate_figure_fourier_window_paths import generate as generate_paths
from generate_figure_fourier_window_sensitivity import generate as generate_sensitivity


INPUT = Path("fourier_artifact_robustness_benchmark")


def test_publication_data_encode_control_sensitivity():
    _payload, _arrays, _strengths, _diagnostics, endpoints, truth, _reference = load_checked_benchmark(INPUT)
    fractional_change = np.abs((endpoints[-1] - truth) / truth)
    assert fractional_change[1] > fractional_change[2] > fractional_change[0]
    assert fractional_change[1] > 0.99


def test_publication_drivers_create_finite_audit_outputs(tmp_path):
    generate_sensitivity(INPUT, tmp_path)
    generate_paths(INPUT, tmp_path)
    expected = (
        "figure_fourier_window_sensitivity.pdf",
        "figure_fourier_window_sensitivity.png",
        "figure_fourier_window_sensitivity.csv",
        "figure_fourier_window_paths.pdf",
        "figure_fourier_window_paths.png",
        "figure_fourier_window_paths.npz",
    )
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
    with np.load(tmp_path / "figure_fourier_window_paths.npz") as data:
        assert all(np.all(np.isfinite(data[key])) for key in data.files)
