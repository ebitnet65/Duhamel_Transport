#!/usr/bin/env python3
"""Headless deterministic synthetic endpoint-recovery benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from dimer_endpoint_inversion import (
    EndpointRecoveryConfig,
    generate_synthetic_truth,
    recover_dimer_endpoint,
)
from dimer_spectral_observable import (
    DimerSpectrumConfig,
    build_packed_dimer_spectral_observable,
)
from runtime_metadata import collect_runtime_metadata, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dimer_endpoint_recovery.json"))
    parser.add_argument("--n-omega", type=int, default=7)
    parser.add_argument("--t2", type=float, default=0.8)
    parser.add_argument("--truth", type=float, nargs=3, default=(0.025, 0.014, 0.022))
    parser.add_argument("--relative-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--random-starts", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = utc_timestamp()
    start_clock = time.perf_counter()
    frequencies = np.linspace(0.86, 1.32, args.n_omega)
    spectrum_config = DimerSpectrumConfig(frequencies, frequencies, eta=0.03)
    observable = build_packed_dimer_spectral_observable(spectrum_config)
    truth = generate_synthetic_truth(
        observable,
        args.truth,
        args.t2,
        relative_noise=args.relative_noise,
        seed=args.seed,
    )
    recovery_config = EndpointRecoveryConfig(
        starts=((0.04, 0.006, 0.012),),
        n_random_starts=args.random_starts,
        seed=args.seed,
    )
    result = recover_dimer_endpoint(
        observable,
        truth.observation,
        truth.t2,
        config=recovery_config,
        truth_endpoint=truth.endpoint,
    )
    payload = {
        "schema_version": 1,
        "control_order": ["J_12", "k_12", "gamma_phi"],
        "frequency_grid": frequencies.tolist(),
        "spectrum_config": {
            "eta": spectrum_config.eta,
            "mean_site_energy": spectrum_config.mean_site_energy,
            "detuning": spectrum_config.detuning,
            "beta": spectrum_config.beta,
        },
        "truth": truth.to_dict(),
        "recovery_config": {
            "bounds": recovery_config.bounds,
            "n_random_starts": recovery_config.n_random_starts,
            "seed": recovery_config.seed,
        },
        "recovery": result.to_dict(),
        "runtime": collect_runtime_metadata(
            command=sys.argv,
            model_metadata={
                "controls": ["J_12", "k_12", "gamma_phi"],
                "n_omega": args.n_omega,
                "t2": args.t2,
            },
            seeds={"inversion": args.seed},
            repo_dir=Path(__file__).resolve().parent,
            started_at=started_at,
            finished_at=utc_timestamp(),
            exit_code=0 if result.success else 1,
        ),
        "wall_seconds": time.perf_counter() - start_clock,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), **result.to_dict()}, indent=2))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
