"""End-to-end spectral observability test on the physical dimer geometry."""

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from dimer_covariant_geometry import build_dimer_geometry_backend  # noqa: E402
from dimer_spectral_observable import (  # noqa: E402
    DimerSpectrumConfig,
    build_packed_dimer_spectral_observable,
)
from spectral_observability import (  # noqa: E402
    generalized_observability_modes,
    information_across_waiting_times,
)


def test_dimer_spectrum_defines_covariant_modes_across_waiting_times():
    frequencies = np.linspace(0.90, 1.28, 3)
    observable = build_packed_dimer_spectral_observable(
        DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    )
    geometry = build_dimer_geometry_backend(
        control_mode="coupling_transfer_dephasing",
        fixed_detuning=0.18,
        beta_12=2.5,
    )
    q = jnp.asarray([0.04, 0.006, 0.012], dtype=jnp.float64)
    waiting_times = (0.4, 1.2)
    jacobians = [
        np.asarray(jax.jacfwd(lambda x: observable(x, t2))(q))
        for t2 in waiting_times
    ]
    joint, individual = information_across_waiting_times(jacobians)
    physical_metric = np.asarray(geometry.metric(q))
    modes = generalized_observability_modes(joint, physical_metric)

    np.testing.assert_allclose(joint, individual[0] + individual[1])
    np.testing.assert_allclose(
        modes.eigenvectors.T @ physical_metric @ modes.eigenvectors,
        np.eye(3),
        rtol=2e-9,
        atol=2e-9,
    )
    assert modes.effective_rank == 3
    assert np.all(np.isfinite(modes.eigenvalues))
    assert np.linalg.norm(individual[1] - individual[0]) > 1e-6
