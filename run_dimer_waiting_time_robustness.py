#!/usr/bin/env python3
"""Deterministic fixed-endpoint recovery and observability sweep over t2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dimer_covariant_geometry import build_dimer_geometry_backend
from dimer_endpoint_inversion import EndpointRecoveryConfig, recover_dimer_endpoint
from dimer_spectral_observable import (
    DimerSpectrumConfig,
    build_packed_dimer_spectral_observable,
)
from waiting_time_robustness import analyze_waiting_time_robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-omega", type=int, default=5)
    parser.add_argument("--t2", type=float, nargs="+", default=(0.2, 0.8, 1.6))
    parser.add_argument("--truth", type=float, nargs=3, default=(0.025, 0.014, 0.022))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--random-starts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_omega < 2:
        raise ValueError("n-omega must be at least two")
    frequencies = np.linspace(0.86, 1.32, args.n_omega)
    spectrum_config = DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    observable = build_packed_dimer_spectral_observable(spectrum_config)
    geometry = build_dimer_geometry_backend(
        control_mode="coupling_transfer_dephasing",
        fixed_detuning=spectrum_config.detuning,
        beta_12=spectrum_config.beta,
    )
    truth = np.asarray(args.truth, dtype=float)
    recovery_config = EndpointRecoveryConfig(
        starts=((0.04, 0.006, 0.012),),
        n_random_starts=args.random_starts,
        seed=args.seed,
    )
    recovery_records = {}

    def recover(waiting_time, truth_observation):
        result = recover_dimer_endpoint(
            observable,
            truth_observation,
            waiting_time,
            config=recovery_config,
            truth_endpoint=truth,
        )
        recovery_records[str(waiting_time)] = result.to_dict()
        return result

    result = analyze_waiting_time_robustness(
        truth,
        args.t2,
        observable,
        recover,
        geometry.metric,
    )
    payload = {
        "schema_version": 1,
        "control_order": ["J_12", "k_12", "gamma_phi"],
        "frequency_grid": frequencies.tolist(),
        "waiting_times": list(map(float, args.t2)),
        "spectrum_config": {
            "eta": spectrum_config.eta,
            "mean_site_energy": spectrum_config.mean_site_energy,
            "detuning": spectrum_config.detuning,
            "beta": spectrum_config.beta,
        },
        "recovery_config": {
            "bounds": recovery_config.bounds,
            "n_random_starts": recovery_config.n_random_starts,
            "seed": recovery_config.seed,
        },
        "analysis": result.to_dict(),
        "recoveries": recovery_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "waiting_time_robustness.json"
    npz_path = args.output_dir / "waiting_time_robustness.npz"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        npz_path,
        truth_endpoint=result.truth_endpoint,
        waiting_times=np.asarray(args.t2, dtype=float),
        recovered_endpoints=np.stack([item.endpoint for item in result.diagnostics]),
        endpoint_drifts=np.asarray([item.endpoint_drift for item in result.diagnostics]),
        relative_residuals=np.asarray(
            [item.relative_residual_norm for item in result.diagnostics]
        ),
        jacobian_condition_numbers=np.asarray(
            [item.jacobian_condition_number for item in result.diagnostics]
        ),
        generalized_eigenvalues=np.stack(
            [item.generalized_eigenvalues for item in result.diagnostics]
        ),
        information_metrics=np.stack(
            [item.information_metric for item in result.diagnostics]
        ),
        joint_information_metric=result.joint_information_metric,
        joint_generalized_eigenvalues=result.joint_generalized_eigenvalues,
    )
    summary = {
        "json": str(json_path),
        "npz": str(npz_path),
        "maximum_endpoint_drift": max(item.endpoint_drift for item in result.diagnostics),
        "maximum_relative_residual": max(
            item.relative_residual_norm for item in result.diagnostics
        ),
        "joint_effective_rank": result.joint_effective_rank,
        "joint_generalized_eigenvalues": result.joint_generalized_eigenvalues.tolist(),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not all(record["success"] for record in recovery_records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
