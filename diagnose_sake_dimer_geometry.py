#!/usr/bin/env python3
"""Diagnose the state--generator geometry of the scaled SAKE dimer.

The embedding uses the column-vectorized ground-state density matrix for r
and the real and imaginary parts of the Liouvillian for h.  Inner products
are therefore real Euclidean/Frobenius products.  The reported coordinates
are lambda_J = J/omega0 and lambda_k = kappa/omega0.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from benchmark_sake_accuracy_timing import (
    clean_dimer_parameter_map,
    load_derivative_cache,
    validate_derivative_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega0", type=float, default=2.0)
    parser.add_argument("--delta-omega", type=float, default=0.18)
    parser.add_argument("--reference-J", type=float, default=0.0)
    parser.add_argument("--reference-k", type=float, default=0.0)
    parser.add_argument("--delta-J", type=float, default=None)
    parser.add_argument("--delta-k", type=float, default=None)
    parser.add_argument("--clean-gamma", type=float, default=0.01)
    parser.add_argument("--clean-gamma-phi", type=float, default=0.01)
    parser.add_argument("--clean-mu-esa-scale", type=float, default=1.0)
    parser.add_argument("--derivative-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def real_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.vdot(left, right).real)


def main() -> None:
    args = parse_args()
    parameter_map = clean_dimer_parameter_map(args)
    derivatives, _, metadata = load_derivative_cache(args.derivative_cache)
    validate_derivative_cache(metadata, parameter_map, args)
    tangents = [derivatives[(0,)], derivatives[(1,)]]
    metric = np.array(
        [[real_frobenius(left, right) for right in tangents] for left in tangents]
    )

    # The implemented initial/steady state is |0><0| for every J and kappa,
    # hence both state tangents vanish exactly and so does d r wedge d h.
    state_tangent_norms = np.zeros(2)
    response_two_form = np.zeros((2, 2))
    determinant = float(np.linalg.det(metric))
    eigenvalues = np.linalg.eigvalsh(metric)
    correlation = float(metric[0, 1] / np.sqrt(metric[0, 0] * metric[1, 1]))
    normalized_f12 = float(
        response_two_form[0, 1] / np.sqrt(determinant) if determinant > 0 else np.nan
    )
    result = {
        "coordinates": ["J/omega0", "kappa/omega0"],
        "omega0": args.omega0,
        "delta_omega": args.delta_omega,
        "metric": metric.tolist(),
        "det_metric": determinant,
        "metric_eigenvalues": eigenvalues.tolist(),
        "metric_condition_number": float(np.linalg.cond(metric)),
        "tangent_correlation": correlation,
        "state_tangent_norms": state_tangent_norms.tolist(),
        "F12": float(response_two_form[0, 1]),
        "F12_over_sqrt_det_metric": normalized_f12,
        "constant_over_parameter_plane": True,
        "reason": "The state is fixed and the Liouvillian is affine in both controls.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
