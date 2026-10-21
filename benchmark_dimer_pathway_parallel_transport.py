#!/usr/bin/env python3
"""End-to-end ordered Liouville-pathway transport to a recovered dimer endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import Liouville_pathway_parallel_transport as pathway_transport
import Liouville_transport_general as general
from benchmark_recovered_endpoint_transport import compare_routes_to_endpoint
from exciton_dimer_model import (
    detailed_balance_site_transfer_rates,
    exciton_dimer_hamiltonian,
    exciton_dimer_mu_components,
    op,
    projector,
)
from fit_v_system_transport_signal import pathway_signal_grid


def _physical_dimer_model_and_map() -> tuple[general.LiouvilleModel, general.ParameterMap]:
    """Return the benchmark model with detailed balance and collective dephasing."""

    base = {
        "omega_1": 1.0,
        "omega_2": 1.18,
        "anharmonicity": -0.06,
        "J_12": 0.0,
        "k_12": 0.0,
        "gamma_phi": 0.0,
        "gamma_1": 0.03,
        "gamma_2": 0.004,
        "beta_12": 2.5,
        "mu_1": 1.0,
        "mu_2": 1.0,
        "mu_esa_scale": 1.0,
    }

    def collapse_ops(params):
        operators = [
            np.sqrt(params["gamma_1"]) * op(0, 1),
            np.sqrt(params["gamma_1"]) * op(2, 3),
            np.sqrt(params["gamma_2"]) * op(0, 2),
            np.sqrt(params["gamma_2"]) * op(1, 3),
        ]
        forward, reverse = detailed_balance_site_transfer_rates(params)
        if forward > 0.0:
            operators.append(np.sqrt(forward) * op(2, 1))
        if reverse > 0.0:
            operators.append(np.sqrt(reverse) * op(1, 2))
        if params["gamma_phi"] > 0.0:
            z_sum = np.diag([-2.0, 0.0, 0.0, 2.0]).astype(complex)
            operators.append(np.sqrt(params["gamma_phi"]) * z_sum)
        return tuple(operators)

    model = general.LiouvilleModel.from_hamiltonian(
        dimension=4,
        hamiltonian=exciton_dimer_hamiltonian,
        collapse_ops=collapse_ops,
        initial_state=lambda _params: general.vec(projector(0)),
        transition_dipole=lambda params: sum(exciton_dimer_mu_components(params)),
        dipole_components=exciton_dimer_mu_components,
    )
    parameter_map = general.ParameterMap(
        reference_params=base,
        controls=(
            general.ControlSpec("J_12", 0.0),
            general.ControlSpec("k_12", 0.0),
            general.ControlSpec("gamma_phi", 0.0),
        ),
    )
    return model, parameter_map


def _ordered_independent_pathways(model, parameter_map, reference, omegas, *, t2, eta):
    selected = []
    for sector in ("RP", "NRP"):
        candidates = general.canonical_pathways((sector,))
        matrix = general.liouville_pathway_basis_matrix(
            model, parameter_map, reference, omegas, candidates,
            stage="after_U2", tau2=t2, eta=eta,
        )
        indices, _, _ = general.select_independent_columns(
            matrix, [item.label for item in candidates], rtol=1.0e-9
        )
        selected.extend(candidates[index] for index in indices)
    return selected


def _route_summary(refinement) -> dict[str, Any]:
    result = refinement.results[-1]
    mixing = np.abs(result.final_transport) ** 2
    off_diagonal = mixing - np.diag(np.diag(mixing))
    return {
        "refinement_factors": list(refinement.refinement_factors),
        "successive_transport_errors": refinement.successive_transport_errors.tolist(),
        "successive_ambient_frame_errors": refinement.successive_ambient_frame_errors.tolist(),
        "maximum_principal_angle": float(np.max(result.principal_angles)),
        "endpoint_subspace_angles": np.arccos(
            np.clip(
                np.linalg.svd(
                    result.orthonormal_frames[-1].conj().T
                    @ result.orthonormal_frames[0],
                    compute_uv=False,
                ),
                0.0,
                1.0,
            )
        ).tolist(),
        "maximum_frame_condition": float(np.max(result.frame_condition_numbers)),
        "maximum_overlap_condition": float(np.max(result.overlap_condition_numbers)),
        "maximum_unitarity_error": float(np.max(result.local_unitarity_errors)),
        "maximum_cumulative_gram_error": float(np.max(result.cumulative_gram_errors)),
        "pathway_mixing_matrix": mixing.tolist(),
        "off_diagonal_mixing_fraction": float(np.sum(off_diagonal) / np.sum(mixing)),
    }


def run_benchmark(
    reference: Sequence[float], q_star: Sequence[float], *, n_vertices: int = 7,
    n_omega: int = 3, t2: float = 0.8, eta: float = 0.03,
    refinement_factors: Sequence[int] = (1, 2, 4),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    route_diagnostics, route_arrays = compare_routes_to_endpoint(
        reference, q_star, n_vertices=n_vertices
    )
    model, parameter_map = _physical_dimer_model_and_map()
    omegas = np.linspace(0.86, 1.32, n_omega)
    pathways = _ordered_independent_pathways(
        model, parameter_map, np.asarray(reference), omegas, t2=t2, eta=eta
    )
    frame_fn = pathway_transport.build_liouville_pathway_frame_function(
        model, parameter_map, omegas, pathways,
        stage="after_U2", tau2=t2, eta=eta,
    )
    straight = pathway_transport.transport_with_refinement(
        route_arrays["straight_path"], frame_fn,
        refinement_factors=refinement_factors,
    )
    geodesic = pathway_transport.transport_with_refinement(
        route_arrays["geodesic_path"], frame_fn,
        refinement_factors=refinement_factors,
    )
    straight_final = straight.results[-1]
    geodesic_final = geodesic.results[-1]
    route_holonomy = geodesic_final.final_transport @ straight_final.final_transport.conj().T
    route_frame_difference = np.linalg.norm(
        geodesic_final.aligned_frames[-1] - straight_final.aligned_frames[-1], ord="fro"
    )

    straight_endpoint_signals = pathway_signal_grid(
        model, parameter_map, route_arrays["straight_path"][-1], omegas, omegas, pathways,
        tau2=t2, eta=eta,
    )
    geodesic_endpoint_signals = pathway_signal_grid(
        model, parameter_map, route_arrays["geodesic_path"][-1], omegas, omegas, pathways,
        tau2=t2, eta=eta,
    )
    straight_endpoint_spectrum = np.sum(straight_endpoint_signals, axis=1).reshape(n_omega, n_omega)
    geodesic_endpoint_spectrum = np.sum(geodesic_endpoint_signals, axis=1).reshape(n_omega, n_omega)
    # This is evaluated independently at each route endpoint, not hard-coded.
    endpoint_route_spectral_difference = float(
        np.linalg.norm(geodesic_endpoint_spectrum - straight_endpoint_spectrum)
    )
    diagnostics = {
        "schema_version": 1,
        "reference": np.asarray(reference, dtype=float).tolist(),
        "q_star": np.asarray(q_star, dtype=float).tolist(),
        "control_order": ["J_12", "k_12", "gamma_phi"],
        "pathway_labels": [item.label for item in pathways],
        "pathway_sectors": [item.sector for item in pathways],
        "n_pathways": len(pathways),
        "n_omega": n_omega,
        "t2": t2,
        "eta": eta,
        "physical_route": route_diagnostics["geodesic"],
        "straight": _route_summary(straight),
        "geodesic": _route_summary(geodesic),
        "route_comparison": {
            "endpoint_spectral_difference": endpoint_route_spectral_difference,
            "endpoint_aligned_frame_frobenius_difference": float(route_frame_difference),
            "holonomy_frobenius_distance_from_identity": float(
                np.linalg.norm(route_holonomy - np.eye(len(pathways)), ord="fro")
            ),
            "holonomy_unitarity_error": float(
                np.linalg.norm(route_holonomy.conj().T @ route_holonomy - np.eye(len(pathways)), ord="fro")
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays = {
        **route_arrays,
        "omega_grid": omegas,
        "straight_endpoint_pathway_signals": straight_endpoint_signals,
        "geodesic_endpoint_pathway_signals": geodesic_endpoint_signals,
        "straight_endpoint_spectrum": straight_endpoint_spectrum,
        "geodesic_endpoint_spectrum": geodesic_endpoint_spectrum,
        "straight_final_transport": straight_final.final_transport,
        "geodesic_final_transport": geodesic_final.final_transport,
        "route_holonomy": route_holonomy,
        "straight_aligned_endpoint_frame": straight_final.aligned_frames[-1],
        "geodesic_aligned_endpoint_frame": geodesic_final.aligned_frames[-1],
    }
    return diagnostics, arrays


def _load_endpoint(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["recovery"]["endpoint"], dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-json", type=Path, required=True)
    parser.add_argument("--reference", nargs=3, type=float, default=(0.070, 0.003, 0.004))
    parser.add_argument("--vertices", type=int, default=7)
    parser.add_argument("--n-omega", type=int, default=3)
    parser.add_argument("--t2", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--refinement-factors", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    diagnostics, arrays = run_benchmark(
        args.reference, _load_endpoint(args.recovery_json), n_vertices=args.vertices,
        n_omega=args.n_omega, t2=args.t2, eta=args.eta,
        refinement_factors=args.refinement_factors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pathway_transport_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.output_dir / "pathway_transport_arrays.npz", **arrays)
    print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
