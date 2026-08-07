#!/usr/bin/env python3
"""Prepare a lightweight JAX derivative cache for the SAKE dimer benchmark.

Running JAX in a short-lived process prevents compiled XLA state from
contributing to the memory footprint of the longer pathway benchmark.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

import Liouville_autodiff as autodiff
import Liouville_transport_general as general
from exciton_dimer_autodiff import jax_exciton_dimer_liouvillian
from exciton_dimer_model import exciton_dimer_model, exciton_dimer_parameter_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega0", type=float, default=2.0)
    parser.add_argument("--delta-omega", type=float, default=0.18)
    parser.add_argument("--reference-J", type=float, default=0.0)
    parser.add_argument("--reference-k", type=float, default=0.0)
    parser.add_argument("--delta-J", type=float, default=None)
    parser.add_argument("--delta-k", type=float, default=None)
    parser.add_argument(
        "--max-order",
        type=int,
        choices=(2, 3),
        default=2,
        help="Order 2 supplies the explicit zero Hessian required by the transport builder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dimer_backend_benchmarks/sake_dimer_derivatives.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.omega0 <= 0.0:
        raise ValueError("omega0 must be positive")
    if abs(args.delta_omega) >= 2.0 * args.omega0:
        raise ValueError("abs(delta_omega) must be smaller than 2*omega0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    model = exciton_dimer_model()
    scale_J = args.omega0 if args.delta_J is None else args.delta_J
    scale_k = args.omega0 if args.delta_k is None else args.delta_k
    reference_params = dict(exciton_dimer_parameter_map().reference_params)
    reference_params.update(
        {
            "omega_1": args.omega0 + 0.5 * args.delta_omega,
            "omega_2": args.omega0 - 0.5 * args.delta_omega,
            "J_12": args.reference_J,
            "k_12": args.reference_k,
            "anharmonicity": 0.0,
            "gamma_1": 0.01,
            "gamma_2": 0.01,
            "gamma_phi_1": 0.01,
            "gamma_phi_2": 0.01,
            "mu_1": 1.0,
            "mu_2": 1.0,
            "mu_esa_scale": 1.0,
        }
    )
    parameter_map = general.ParameterMap(
        reference_params=reference_params,
        controls=(
            general.ControlSpec("J_12", args.reference_J, scale_J),
            general.ControlSpec("k_12", args.reference_k, scale_k),
        ),
    )
    start = time.perf_counter()
    derivatives = autodiff.jax_liouvillian_derivatives(
        jax_exciton_dimer_liouvillian,
        parameter_map,
        liouville_dimension=model.liouville_dimension,
        max_order=args.max_order,
    )
    derivative_seconds = time.perf_counter() - start
    arrays = {
        ("root" if not key else "d_" + "_".join(str(index) for index in key)): value
        for key, value in derivatives.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "derivative_seconds": derivative_seconds,
        "max_order": args.max_order,
        "omega0": args.omega0,
        "delta_omega": args.delta_omega,
        "omega_1": args.omega0 + 0.5 * args.delta_omega,
        "omega_2": args.omega0 - 0.5 * args.delta_omega,
        "reference_J": args.reference_J,
        "reference_k": args.reference_k,
        "scale_J": scale_J,
        "scale_k": scale_k,
        "python": sys.version,
        "platform": platform.platform(),
        "keys": [list(key) for key in derivatives],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
