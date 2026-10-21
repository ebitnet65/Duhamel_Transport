"""Controlled time-to-frequency artifacts for dimer model-discrepancy tests.

The source spectrum is interpreted as samples on uniform ascending angular-
frequency grids.  After removing each carrier frequency, the centered unitary
discrete transform pair is

    R(t3,t1) = IFFT2[S(Omega3,Omega1)],
    S(Omega3,Omega1) = FFT2[R(t3,t1)].

The array convention is always ``(axis3, axis1)``.  Acquisition artifacts are
applied in the time domain before the forward transform.  A timing-origin
offset is represented after transformation by

    exp[-i (omega1 delta_t1 + omega3 delta_t3)],

with an optional global heterodyne phase.  This layer changes the observable,
not the fixed endpoint ``q_true`` used to generate the source spectrum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class AcquisitionArtifactConfig:
    """Deterministic acquisition and Fourier-processing controls.

    Tuple ordering is ``(axis3, axis1)``.  Apodization strengths are inverse
    time scales: exponential uses ``exp(-a |t|)`` and Gaussian uses
    ``exp(-(a t)^2/2)``.
    """

    window_fraction: tuple[float, float] = (1.0, 1.0)
    apodization: str = "none"
    apodization_strength: tuple[float, float] = (0.0, 0.0)
    sample_stride: tuple[int, int] = (1, 1)
    zero_pad_factor: tuple[int, int] = (1, 1)
    global_phase: float = 0.0
    time_zero_shift: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class AcquisitionTruthResult:
    """Ideal source and distorted observation with auditable grids."""

    source_ideal_spectrum: Array
    reconstructed_ideal_spectrum: Array
    distorted_spectrum: Array
    source_time_signal: Array
    acquired_time_signal: Array
    omega1_source: Array
    omega3_source: Array
    omega1_grid: Array
    omega3_grid: Array
    t1_source: Array
    t3_source: Array
    t1_acquired: Array
    t3_acquired: Array
    artifact_config: AcquisitionArtifactConfig
    roundtrip_relative_error: float
    retained_time_energy_fraction: float
    metadata: dict[str, Any]

    def artifact_metadata(self) -> dict[str, Any]:
        """Return JSON-safe acquisition metadata without large arrays."""

        return {
            **self.metadata,
            "artifact_config": asdict(self.artifact_config),
            "roundtrip_relative_error": self.roundtrip_relative_error,
            "retained_time_energy_fraction": self.retained_time_energy_fraction,
            "source_shape": list(self.source_ideal_spectrum.shape),
            "output_shape": list(self.distorted_spectrum.shape),
            "omega1_output": self.omega1_grid.tolist(),
            "omega3_output": self.omega3_grid.tolist(),
        }


def _uniform_grid(values: Sequence[float], name: str) -> tuple[Array, float, float]:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a finite one-dimensional grid of length >= 2")
    differences = np.diff(grid)
    if np.any(differences <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    spacing = float(np.mean(differences))
    if not np.allclose(differences, spacing, rtol=1e-10, atol=1e-13):
        raise ValueError(f"{name} must be uniformly spaced")
    center = float(0.5 * (grid[0] + grid[-1]))
    expected = center + (np.arange(grid.size) - (grid.size - 1) / 2.0) * spacing
    if not np.allclose(grid, expected, rtol=1e-10, atol=1e-13):
        raise ValueError(f"{name} must be centered on its midpoint")
    return grid, spacing, center


def _time_grid(size: int, angular_frequency_spacing: float) -> Array:
    return np.fft.fftshift(
        np.fft.fftfreq(size, d=angular_frequency_spacing / (2.0 * np.pi))
    )


def _centered_indices(size: int, stride: int) -> Array:
    start = (size // 2) % stride
    return np.arange(start, size, stride, dtype=int)


def _center_pad(values: Array, output_shape: tuple[int, int]) -> Array:
    padding = []
    for old, new in zip(values.shape, output_shape, strict=True):
        if new < old:
            raise ValueError("output_shape cannot be smaller than sampled data")
        before = new // 2 - old // 2
        padding.append((before, new - old - before))
    return np.pad(values, padding, mode="constant")


def _validate_config(config: AcquisitionArtifactConfig) -> None:
    if config.apodization not in {"none", "exponential", "gaussian"}:
        raise ValueError("apodization must be none, exponential, or gaussian")
    if any(not 0.0 < value <= 1.0 for value in config.window_fraction):
        raise ValueError("window fractions must lie in (0, 1]")
    if any(value < 0.0 for value in config.apodization_strength):
        raise ValueError("apodization strengths must be nonnegative")
    if any(int(value) != value or value < 1 for value in config.sample_stride):
        raise ValueError("sample strides must be positive integers")
    if any(int(value) != value or value < 1 for value in config.zero_pad_factor):
        raise ValueError("zero-pad factors must be positive integers")


def generate_acquisition_truth(
    ideal_spectrum: Array,
    omega1_grid: Sequence[float],
    omega3_grid: Sequence[float],
    *,
    config: AcquisitionArtifactConfig = AcquisitionArtifactConfig(),
) -> AcquisitionTruthResult:
    """Apply deterministic acquisition artifacts to an ideal complex spectrum."""

    _validate_config(config)
    omega1, domega1, _midpoint1 = _uniform_grid(omega1_grid, "omega1_grid")
    omega3, domega3, _midpoint3 = _uniform_grid(omega3_grid, "omega3_grid")
    spectrum = np.asarray(ideal_spectrum, dtype=complex)
    if spectrum.shape != (omega3.size, omega1.size):
        raise ValueError("ideal_spectrum must have shape (len(omega3), len(omega1))")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("ideal_spectrum contains non-finite values")

    source_time = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho")
    )
    reconstructed = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(source_time), norm="ortho")
    )
    spectrum_norm = max(np.linalg.norm(spectrum), np.finfo(float).tiny)
    roundtrip_error = float(np.linalg.norm(reconstructed - spectrum) / spectrum_norm)
    t1 = _time_grid(omega1.size, domega1)
    t3 = _time_grid(omega3.size, domega3)
    # Even-length FFT grids are asymmetric around zero.  Choose the discrete
    # carrier that makes the artifact-free output reproduce the supplied grid
    # exactly rather than forcing the arithmetic midpoint as carrier.
    source_detuning1 = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(omega1.size, d=float(t1[1] - t1[0]))
    )
    source_detuning3 = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(omega3.size, d=float(t3[1] - t3[0]))
    )
    carrier1 = float(omega1[0] - source_detuning1[0])
    carrier3 = float(omega3[0] - source_detuning3[0])

    max_t1 = max(float(np.max(np.abs(t1))), np.finfo(float).tiny)
    max_t3 = max(float(np.max(np.abs(t3))), np.finfo(float).tiny)
    window1 = np.abs(t1) <= config.window_fraction[1] * max_t1 + 1e-14
    window3 = np.abs(t3) <= config.window_fraction[0] * max_t3 + 1e-14
    window = np.outer(window3, window1)
    a3, a1 = config.apodization_strength
    if config.apodization == "none":
        apodization = np.ones_like(window, dtype=float)
    elif config.apodization == "exponential":
        apodization = np.exp(-a3 * np.abs(t3)[:, None] - a1 * np.abs(t1)[None, :])
    else:
        apodization = np.exp(
            -0.5 * (a3 * t3[:, None]) ** 2 - 0.5 * (a1 * t1[None, :]) ** 2
        )
    processed = source_time * window * apodization
    source_energy = max(float(np.vdot(source_time, source_time).real), np.finfo(float).tiny)
    retained_energy = float(np.vdot(processed, processed).real / source_energy)

    stride3, stride1 = map(int, config.sample_stride)
    indices3 = _centered_indices(t3.size, stride3)
    indices1 = _centered_indices(t1.size, stride1)
    if indices3.size < 2 or indices1.size < 2:
        raise ValueError("sample_stride must retain at least two points on each axis")
    sampled = processed[np.ix_(indices3, indices1)]
    t3_sampled = t3[indices3]
    t1_sampled = t1[indices1]
    dt3 = float(t3_sampled[1] - t3_sampled[0])
    dt1 = float(t1_sampled[1] - t1_sampled[0])

    pad3, pad1 = map(int, config.zero_pad_factor)
    output_shape = (sampled.shape[0] * pad3, sampled.shape[1] * pad1)
    acquired_time = _center_pad(sampled, output_shape)
    distorted = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(acquired_time), norm="ortho")
    )
    # Preserve spectral amplitude when zero padding under the unitary FFT.
    distorted *= np.sqrt(acquired_time.size / sampled.size)
    detuning3 = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(output_shape[0], d=dt3))
    detuning1 = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(output_shape[1], d=dt1))
    omega3_output = carrier3 + detuning3
    omega1_output = carrier1 + detuning1

    shift3, shift1 = config.time_zero_shift
    phase = np.exp(
        1j * config.global_phase
        - 1j
        * (
            omega3_output[:, None] * shift3
            + omega1_output[None, :] * shift1
        )
    )
    distorted = distorted * phase
    metadata = {
        "axis_order": ["omega3", "omega1"],
        "transform": "centered numpy FFT2/IFFT2 with norm='ortho'",
        "timing_phase_sign": "exp[-i(omega3*delta_t3 + omega1*delta_t1)]",
        "carrier_frequencies": [carrier3, carrier1],
        "source_angular_frequency_spacings": [domega3, domega1],
        "sampled_time_spacings": [dt3, dt1],
        "sampled_shape_before_padding": list(sampled.shape),
    }
    return AcquisitionTruthResult(
        source_ideal_spectrum=spectrum,
        reconstructed_ideal_spectrum=reconstructed,
        distorted_spectrum=distorted,
        source_time_signal=source_time,
        acquired_time_signal=acquired_time,
        omega1_source=omega1,
        omega3_source=omega3,
        omega1_grid=omega1_output,
        omega3_grid=omega3_output,
        t1_source=t1,
        t3_source=t3,
        t1_acquired=(np.arange(output_shape[1]) - output_shape[1] // 2) * dt1,
        t3_acquired=(np.arange(output_shape[0]) - output_shape[0] // 2) * dt3,
        artifact_config=config,
        roundtrip_relative_error=roundtrip_error,
        retained_time_energy_fraction=retained_energy,
        metadata=metadata,
    )


def generate_dimer_acquisition_truth(
    complex_observable: Callable[[Array, float], Array],
    q_true: Sequence[float],
    t2: float,
    omega1_grid: Sequence[float],
    omega3_grid: Sequence[float],
    *,
    config: AcquisitionArtifactConfig = AcquisitionArtifactConfig(),
) -> AcquisitionTruthResult:
    """Evaluate the fixed endpoint then apply the controlled acquisition layer."""

    ideal = np.asarray(complex_observable(np.asarray(q_true, dtype=float), float(t2)))
    return generate_acquisition_truth(
        ideal, omega1_grid, omega3_grid, config=config
    )


__all__ = [
    "AcquisitionArtifactConfig",
    "AcquisitionTruthResult",
    "generate_acquisition_truth",
    "generate_dimer_acquisition_truth",
]
