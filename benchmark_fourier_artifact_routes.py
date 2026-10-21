#!/usr/bin/env python3
"""Fourier-artifact endpoint recovery, geodesics, and pathway transport sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark_dimer_pathway_parallel_transport import run_benchmark as run_pathway
from dimer_acquisition_truth import AcquisitionArtifactConfig, generate_dimer_acquisition_truth
from dimer_covariant_geometry import build_dimer_geometry_backend
from dimer_endpoint_inversion import EndpointRecoveryConfig, recover_dimer_endpoint
from dimer_spectral_observable import DimerSpectrumConfig, build_dimer_spectral_observable, build_packed_dimer_spectral_observable
from model_discrepancy_robustness import ReliabilityThresholds, analyze_model_discrepancy_sweep


def run_artifact_route_sweep(
    strengths: Sequence[float], *, n_omega: int = 5, t2: float = 0.8,
    truth=(0.025, 0.014, 0.022), reference=(0.070, 0.003, 0.004),
    vertices: int = 5, refinement_factors=(1, 2),
):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    strengths = np.asarray(strengths, dtype=float)
    omega = np.linspace(0.86, 1.32, n_omega)
    config = DimerSpectrumConfig(omega, omega, eta=0.03)
    complex_observable = build_dimer_spectral_observable(config)
    packed_observable = build_packed_dimer_spectral_observable(config)
    truth = np.asarray(truth, dtype=float)
    reference = np.asarray(reference, dtype=float)
    geometry = build_dimer_geometry_backend(control_mode="coupling_transfer_dephasing")

    def truth_generator(strength, _seed):
        acquisition = generate_dimer_acquisition_truth(
            complex_observable, truth, t2, omega, omega,
            config=AcquisitionArtifactConfig(
                apodization="exponential" if strength > 0 else "none",
                apodization_strength=(strength, strength),
            ),
        )
        values = acquisition.distorted_spectrum.reshape(-1)
        return np.concatenate((values.real, values.imag))

    recovery_config = EndpointRecoveryConfig(
        starts=(tuple(reference),), n_random_starts=0, seed=0,
    )
    def recover(observed):
        return recover_dimer_endpoint(
            packed_observable, observed, t2, config=recovery_config,
            truth_endpoint=truth,
        )
    def jacobian(endpoint):
        return np.asarray(jax.jacfwd(lambda q: packed_observable(q, t2))(jnp.asarray(endpoint)))

    discrepancy = analyze_model_discrepancy_sweep(
        truth, strengths, truth_generator,
        lambda q: np.asarray(packed_observable(q, t2)), recover, jacobian,
        geometry.metric,
        thresholds=ReliabilityThresholds(
            maximum_relative_residual=0.10,
            # Direction alone is not a failure: the normalized orthogonal
            # magnitude below supplies the quantitative model-error gate.
            maximum_orthogonal_residual_fraction=1.0,
            maximum_euclidean_endpoint_bias=0.02,
            maximum_physical_endpoint_bias=0.5,
            maximum_jacobian_condition_number=1.0e5,
        ),
    )
    route_records = []
    arrays = {"strengths": strengths, "truth": truth, "reference": reference}
    for index, diagnostic in enumerate(discrepancy.diagnostics):
        route, route_arrays = run_pathway(
            reference, diagnostic.endpoint, n_vertices=vertices,
            n_omega=n_omega, t2=t2, refinement_factors=refinement_factors,
        )
        route_records.append({
            "artifact_strength": diagnostic.artifact_strength,
            "endpoint": diagnostic.endpoint.tolist(),
            "physical_route": route["physical_route"],
            "pathway_route_comparison": route["route_comparison"],
            "straight_pathway": route["straight"],
            "geodesic_pathway": route["geodesic"],
        })
        arrays[f"endpoint_{index}"] = diagnostic.endpoint
        arrays[f"covariance_{index}"] = diagnostic.local_covariance
        arrays[f"straight_path_{index}"] = route_arrays["straight_path"]
        arrays[f"geodesic_path_{index}"] = route_arrays["geodesic_path"]
        arrays[f"holonomy_{index}"] = route_arrays["route_holonomy"]
    payload = {
        "schema_version": 1,
        "artifact_family": "symmetric_exponential_apodization",
        "artifact_strength_units": "inverse model time",
        "frequency_grid": omega.tolist(),
        "t2": t2,
        "discrepancy": discrepancy.to_dict(),
        "routes": route_records,
    }
    return payload, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strengths", nargs="+", type=float, default=(0.0, 0.0005, 0.001, 0.002, 0.005, 0.01))
    parser.add_argument("--n-omega", type=int, default=5)
    parser.add_argument("--vertices", type=int, default=5)
    parser.add_argument("--refinement-factors", nargs="+", type=int, default=(1, 2))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload, arrays = run_artifact_route_sweep(
        args.strengths, n_omega=args.n_omega, vertices=args.vertices,
        refinement_factors=args.refinement_factors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fourier_artifact_routes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "fourier_artifact_routes.npz", **arrays)
    print(json.dumps({"output_dir": str(args.output_dir), "uncertainty_window": payload["discrepancy"]["uncertainty_window"]}, indent=2))


if __name__ == "__main__": main()
