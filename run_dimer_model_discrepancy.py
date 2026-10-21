#!/usr/bin/env python3
"""Noise-free dimer endpoint-recovery sweep under acquisition artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dimer_acquisition_truth import (
    AcquisitionArtifactConfig,
    generate_dimer_acquisition_truth,
)
from dimer_covariant_geometry import build_dimer_geometry_backend
from dimer_endpoint_inversion import EndpointRecoveryConfig, recover_dimer_endpoint
from dimer_spectral_observable import (
    DimerSpectrumConfig,
    build_dimer_spectral_observable,
    pack_complex_spectrum,
)
from model_discrepancy_robustness import (
    ReliabilityThresholds,
    analyze_model_discrepancy_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-omega", type=int, default=5)
    parser.add_argument("--t2", type=float, default=0.8)
    parser.add_argument("--truth", type=float, nargs=3, default=(0.025, 0.014, 0.022))
    parser.add_argument(
        "--artifact-strength", type=float, nargs="+", default=(0.0, 0.002, 0.005, 0.01, 0.02)
    )
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--max-relative-residual", type=float, default=0.05)
    parser.add_argument("--max-relative-orthogonal-residual", type=float, default=0.02)
    parser.add_argument("--max-orthogonal-fraction", type=float, default=1.0)
    parser.add_argument("--max-endpoint-bias", type=float, default=0.005)
    parser.add_argument("--max-physical-bias", type=float, default=0.25)
    parser.add_argument("--max-jacobian-condition", type=float, default=1.0e6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_omega < 2:
        raise ValueError("n-omega must be at least two")
    frequencies = np.linspace(0.86, 1.32, args.n_omega)
    spectrum_config = DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    complex_observable = build_dimer_spectral_observable(spectrum_config)

    def ideal_observable(q):
        return np.asarray(pack_complex_spectrum(complex_observable(q, args.t2)))

    def inversion_observable(q, _t2):
        return pack_complex_spectrum(complex_observable(q, args.t2))

    truth = np.asarray(args.truth, dtype=float)
    recovery_config = EndpointRecoveryConfig(
        starts=((0.04, 0.006, 0.012),),
        n_random_starts=args.random_starts,
        seed=args.seed,
    )
    recovery_records = {}
    artifact_records = {}

    def truth_generator(strength, seed):
        if seed is not None:
            raise ValueError("this driver is the deterministic noise-free sweep")
        artifact = generate_dimer_acquisition_truth(
            complex_observable,
            truth,
            args.t2,
            frequencies,
            frequencies,
            config=AcquisitionArtifactConfig(
                apodization="exponential",
                apodization_strength=(strength, strength),
            ),
        )
        artifact_records[str(strength)] = artifact.artifact_metadata()
        return np.asarray(pack_complex_spectrum(artifact.distorted_spectrum))

    def recover(observed):
        # Identify the current strength deterministically by insertion order.
        strength = list(artifact_records)[-1]
        result = recover_dimer_endpoint(
            inversion_observable,
            observed,
            args.t2,
            config=recovery_config,
            truth_endpoint=truth,
        )
        recovery_records[strength] = result.to_dict()
        return result

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def jacobian(q):
        return np.asarray(
            jax.jacfwd(lambda point: pack_complex_spectrum(complex_observable(point, args.t2)))(
                jnp.asarray(q, dtype=jnp.float64)
            )
        )

    geometry = build_dimer_geometry_backend(
        control_mode="coupling_transfer_dephasing",
        fixed_detuning=spectrum_config.detuning,
        beta_12=spectrum_config.beta,
    )
    thresholds = ReliabilityThresholds(
        maximum_relative_residual=args.max_relative_residual,
        maximum_relative_orthogonal_residual=args.max_relative_orthogonal_residual,
        maximum_orthogonal_residual_fraction=args.max_orthogonal_fraction,
        maximum_euclidean_endpoint_bias=args.max_endpoint_bias,
        maximum_physical_endpoint_bias=args.max_physical_bias,
        maximum_jacobian_condition_number=args.max_jacobian_condition,
    )
    result = analyze_model_discrepancy_sweep(
        truth,
        args.artifact_strength,
        truth_generator,
        ideal_observable,
        recover,
        jacobian,
        geometry.metric,
        thresholds=thresholds,
    )

    payload = {
        "schema_version": 1,
        "control_order": ["J_12", "k_12", "gamma_phi"],
        "frequency_grid": frequencies.tolist(),
        "waiting_time": args.t2,
        "truth_endpoint": truth.tolist(),
        "artifact_family": "symmetric exponential apodization",
        "artifact_records": artifact_records,
        "recovery_config": {
            "bounds": recovery_config.bounds,
            "n_random_starts": recovery_config.n_random_starts,
            "seed": recovery_config.seed,
        },
        "recoveries": recovery_records,
        "robustness": result.to_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "model_discrepancy_robustness.json"
    npz_path = args.output_dir / "model_discrepancy_robustness.npz"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        npz_path,
        artifact_strengths=result.artifact_strengths,
        endpoints=np.stack([item.endpoint for item in result.diagnostics]),
        endpoint_biases=np.asarray([item.endpoint_bias for item in result.diagnostics]),
        physical_endpoint_biases=np.asarray(
            [item.physical_endpoint_bias for item in result.diagnostics]
        ),
        relative_residuals=np.asarray([item.relative_residual for item in result.diagnostics]),
        parallel_residual_norms=np.asarray(
            [item.tangent_parallel_residual_norm for item in result.diagnostics]
        ),
        orthogonal_residual_norms=np.asarray(
            [item.tangent_orthogonal_residual_norm for item in result.diagnostics]
        ),
        relative_orthogonal_residuals=np.asarray(
            [item.relative_orthogonal_residual for item in result.diagnostics]
        ),
        jacobian_condition_numbers=np.asarray(
            [item.jacobian_condition_number for item in result.diagnostics]
        ),
        generalized_eigenvalues=np.stack(
            [item.generalized_eigenvalues for item in result.diagnostics]
        ),
        physical_uncertainty_rms=np.asarray(
            [item.physical_uncertainty_rms for item in result.diagnostics]
        ),
        reliable=np.asarray([item.reliable for item in result.diagnostics]),
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "npz": str(npz_path),
                "uncertainty_window": result.uncertainty_window,
                "reliable": [item.reliable for item in result.diagnostics],
                "failure_reasons": [item.failure_reasons for item in result.diagnostics],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
