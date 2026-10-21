import json

import numpy as np

from dimer_acquisition_truth import (
    AcquisitionArtifactConfig,
    generate_acquisition_truth,
    generate_dimer_acquisition_truth,
)
from dimer_spectral_observable import (
    DimerSpectrumConfig,
    build_dimer_spectral_observable,
)


def test_zero_artifact_roundtrip_matches_ideal_spectrum_and_grids():
    omega1 = np.linspace(0.85, 1.33, 9)
    omega3 = np.linspace(0.88, 1.28, 9)
    rows, columns = np.meshgrid(omega3, omega1, indexing="ij")
    ideal = np.exp(-((columns - 1.08) / 0.09) ** 2 - ((rows - 1.10) / 0.08) ** 2)
    ideal = ideal * np.exp(0.2j * columns - 0.1j * rows)
    result = generate_acquisition_truth(ideal, omega1, omega3)
    np.testing.assert_allclose(result.distorted_spectrum, ideal, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result.reconstructed_ideal_spectrum, ideal, atol=2e-14)
    np.testing.assert_allclose(result.omega1_grid, omega1, atol=2e-14)
    np.testing.assert_allclose(result.omega3_grid, omega3, atol=2e-14)
    assert result.roundtrip_relative_error < 2e-15
    assert abs(result.retained_time_energy_fraction - 1.0) < 2e-15


def test_fft_sign_axis_and_unitary_normalization_are_deterministic():
    omega1 = np.linspace(-2.0, 2.0, 9)
    omega3 = np.linspace(-3.0, 3.0, 9)
    ideal = np.zeros((9, 9), dtype=complex)
    ideal[6, 7] = 2.0 - 0.5j
    first = generate_acquisition_truth(ideal, omega1, omega3)
    second = generate_acquisition_truth(ideal, omega1, omega3)
    np.testing.assert_array_equal(first.source_time_signal, second.source_time_signal)
    np.testing.assert_allclose(first.distorted_spectrum, ideal, atol=2e-15)
    np.testing.assert_allclose(
        np.linalg.norm(first.source_time_signal), np.linalg.norm(ideal), atol=2e-15
    )
    # Positive detuning produces positive time-domain phase under IFFT.
    center3, center1 = 4, 4
    ratio = first.source_time_signal[center3, center1 + 1] / first.source_time_signal[center3, center1]
    expected = np.exp(1j * (omega1[7] - omega1[4]) * first.t1_source[center1 + 1])
    np.testing.assert_allclose(ratio, expected, atol=2e-14)


def test_even_length_fft_grid_preserves_supplied_frequency_origin():
    omega1 = np.linspace(0.8, 1.4, 8)
    omega3 = np.linspace(0.9, 1.3, 6)
    ideal = np.arange(48, dtype=float).reshape(6, 8).astype(complex)
    result = generate_acquisition_truth(ideal, omega1, omega3)
    np.testing.assert_allclose(result.omega1_grid, omega1, atol=2e-14)
    np.testing.assert_allclose(result.omega3_grid, omega3, atol=2e-14)
    np.testing.assert_allclose(result.distorted_spectrum, ideal, atol=3e-14)


def test_window_and_apodization_strength_reduce_retained_time_energy_monotonically():
    rng = np.random.default_rng(12)
    ideal = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    omega = np.linspace(0.8, 1.4, 9)
    full = generate_acquisition_truth(ideal, omega, omega)
    windowed = generate_acquisition_truth(
        ideal,
        omega,
        omega,
        config=AcquisitionArtifactConfig(window_fraction=(0.7, 0.7)),
    )
    strongly_windowed = generate_acquisition_truth(
        ideal,
        omega,
        omega,
        config=AcquisitionArtifactConfig(window_fraction=(0.35, 0.35)),
    )
    weak_apodization = generate_acquisition_truth(
        ideal,
        omega,
        omega,
        config=AcquisitionArtifactConfig(
            apodization="exponential", apodization_strength=(0.01, 0.01)
        ),
    )
    strong_apodization = generate_acquisition_truth(
        ideal,
        omega,
        omega,
        config=AcquisitionArtifactConfig(
            apodization="exponential", apodization_strength=(0.04, 0.04)
        ),
    )
    assert full.retained_time_energy_fraction > windowed.retained_time_energy_fraction
    assert windowed.retained_time_energy_fraction > strongly_windowed.retained_time_energy_fraction
    assert weak_apodization.retained_time_energy_fraction > strong_apodization.retained_time_energy_fraction


def test_sampling_padding_phase_and_timing_metadata_are_consistent():
    omega = np.linspace(0.8, 1.4, 9)
    ideal = np.ones((9, 9), dtype=complex)
    config = AcquisitionArtifactConfig(
        sample_stride=(2, 2),
        zero_pad_factor=(3, 2),
        global_phase=0.2,
        time_zero_shift=(0.3, -0.1),
    )
    result = generate_acquisition_truth(ideal, omega, omega, config=config)
    assert result.distorted_spectrum.shape == (15, 10)
    assert result.omega3_grid.size == 15
    assert result.omega1_grid.size == 10
    assert np.all(np.diff(result.omega1_grid) > 0.0)
    assert np.all(np.diff(result.omega3_grid) > 0.0)
    metadata = result.artifact_metadata()
    assert metadata["axis_order"] == ["omega3", "omega1"]
    json.dumps(metadata, allow_nan=False)


def test_fixed_dimer_endpoint_wrapper_changes_observation_not_endpoint():
    omega = np.linspace(0.86, 1.32, 5)
    observable = build_dimer_spectral_observable(
        DimerSpectrumConfig(omega, omega, eta=0.03)
    )
    q_true = np.asarray([0.025, 0.014, 0.022])
    result = generate_dimer_acquisition_truth(
        observable,
        q_true,
        0.8,
        omega,
        omega,
        config=AcquisitionArtifactConfig(
            apodization="gaussian",
            apodization_strength=(0.015, 0.02),
            global_phase=0.1,
        ),
    )
    expected = np.asarray(observable(q_true, 0.8))
    np.testing.assert_allclose(result.source_ideal_spectrum, expected)
    assert np.linalg.norm(result.distorted_spectrum - expected) > 1e-3
