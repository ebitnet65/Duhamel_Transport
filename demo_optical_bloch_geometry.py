"""Diagnostic driver for the optical Bloch geometry test branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import optical_bloch_geometry as obg


def _relative_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    denom = max(1.0, np.linalg.norm(expected))
    return float(np.linalg.norm(actual - expected) / denom)


def run_geometry_demo(lam, output_dir: Path, finite_difference_step: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    fd = obg.finite_difference_geometry(lam, step=finite_difference_step)
    summary = {
        "controls": {
            "Omega": float(lam[0]),
            "Delta": float(lam[1]),
            "gamma": float(lam[2]),
        },
        "finite_difference_step": finite_difference_step,
        "closed_form_stationary_bloch": fd.r.tolist(),
        "closed_form_generator_coordinates": fd.h.tolist(),
        "finite_difference_metric": fd.metric.tolist(),
        "finite_difference_two_form": fd.two_form.tolist(),
        "finite_difference_christoffel": fd.christoffel.tolist(),
    }

    if obg.is_sympy_available():
        symbolic = obg.symbolic_geometry(lam)
        summary["sympy_available"] = True
        summary["symbolic_metric"] = symbolic.metric.tolist()
        summary["symbolic_two_form"] = symbolic.two_form.tolist()
        summary["symbolic_christoffel"] = symbolic.christoffel.tolist()
        summary["symbolic_reference_errors"] = {
            "finite_difference_vs_symbolic_metric": _relative_error(
                fd.metric, symbolic.metric
            ),
            "finite_difference_vs_symbolic_two_form": _relative_error(
                fd.two_form, symbolic.two_form
            ),
            "finite_difference_vs_symbolic_christoffel": _relative_error(
                fd.christoffel, symbolic.christoffel
            ),
        }
    else:
        summary["sympy_available"] = False

    if obg.is_jax_available():
        analytic_ad = obg.jax_analytic_geometry(lam)
        liouvillian_ad = obg.jax_liouvillian_geometry(lam)
        summary["jax_available"] = True
        summary["autodiff_errors"] = {
            "liouvillian_vs_closed_form_r": _relative_error(
                liouvillian_ad["r"], analytic_ad["r"]
            ),
            "liouvillian_vs_closed_form_metric": _relative_error(
                liouvillian_ad["metric"], analytic_ad["metric"]
            ),
            "liouvillian_vs_closed_form_two_form": _relative_error(
                liouvillian_ad["two_form"], analytic_ad["two_form"]
            ),
            "liouvillian_vs_closed_form_christoffel": _relative_error(
                liouvillian_ad["christoffel"], analytic_ad["christoffel"]
            ),
            "finite_difference_vs_closed_form_metric": _relative_error(
                fd.metric, analytic_ad["metric"]
            ),
            "finite_difference_vs_closed_form_two_form": _relative_error(
                fd.two_form, analytic_ad["two_form"]
            ),
            "finite_difference_vs_closed_form_christoffel": _relative_error(
                fd.christoffel, analytic_ad["christoffel"]
            ),
        }
    else:
        summary["jax_available"] = False

    np.savez(
        output_dir / "optical_bloch_geometry.npz",
        lam=np.asarray(lam),
        r=fd.r,
        h=fd.h,
        metric=fd.metric,
        two_form=fd.two_form,
        christoffel=fd.christoffel,
    )
    with (output_dir / "geometry_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate optical Bloch physical geometry diagnostics."
    )
    parser.add_argument("--omega", type=float, default=0.35)
    parser.add_argument("--delta", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--fd-step", type=float, default=3.0e-5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optical_bloch_geometry_demo"),
    )
    args = parser.parse_args()

    lam = np.asarray([args.omega, args.delta, args.gamma], dtype=float)
    summary = run_geometry_demo(lam, args.output_dir, args.fd_step)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
