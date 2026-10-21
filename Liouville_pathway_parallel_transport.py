"""Discrete parallel transport of local Liouville pathway subspaces.

At each path vertex the caller supplies an ambient complex matrix whose
columns are ordered local pathway vectors.  The columns are orthonormalized by
the symmetric (SVD/Löwdin) polar factor.  Between adjacent subspaces, the
coefficient-space projection is

    M_i = Q_{i+1}† Q_i.

Its unitary polar factor is used as the norm-preserving local transport map.
This is the standard closest-unitary/Procrustes convention.  It removes local
phase and unitary-basis ambiguity while retaining genuine subspace rotation.
The path-ordered product maps coefficients in the initial local gauge into the
final local gauge.

Gauge convention
----------------
If ``Q_i -> Q_i U_i`` for arbitrary unitary local gauges, the transport obeys
``T -> U_final^dagger T U_initial``.  Consequently the ambient transported
frame ``Q_final T`` is gauge invariant once the initial physical frame is
fixed.  Individual coefficient matrices are not gauge-invariant observables.

The construction requires constant, full column rank along the path and
nonorthogonal adjacent subspaces.  Rank loss and principal angles approaching
pi/2 are reported and rejected when they cross ``rcond``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray
FrameFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class PathwayFrameTransportResult:
    """Transport maps, aligned frames, and local stability diagnostics."""

    path: Array
    raw_frames: Array
    orthonormal_frames: Array
    aligned_frames: Array
    local_transport_maps: Array
    cumulative_transport_maps: Array
    local_projection_maps: Array
    cumulative_projection_maps: Array
    frame_singular_values: Array
    frame_ranks: Array
    frame_condition_numbers: Array
    subspace_singular_values: Array
    principal_angles: Array
    overlap_ranks: Array
    overlap_condition_numbers: Array
    local_unitarity_errors: Array
    cumulative_gram_errors: Array
    projection_gram_errors: Array
    gauge_alignment_residuals: Array

    @property
    def final_transport(self) -> Array:
        return self.cumulative_transport_maps[-1]

    @property
    def final_projection(self) -> Array:
        return self.cumulative_projection_maps[-1]


@dataclass(frozen=True)
class PathwayFrameRefinementResult:
    """Transport solutions on refined paths and successive convergence errors."""

    refinement_factors: tuple[int, ...]
    results: tuple[PathwayFrameTransportResult, ...]
    successive_transport_errors: Array
    successive_ambient_frame_errors: Array


def build_liouville_pathway_frame_function(
    model: Any,
    parameter_map: Any,
    omegas: Array,
    pathways: Sequence[Any],
    *,
    stage: str = "after_U2",
    tau2: float = 1.0,
    eta: float = 1.0e-3,
    quadrature_weight: bool = True,
) -> FrameFunction:
    """Adapt existing ordered canonical pathways into a local-frame callback.

    The ambient rows are frequency-stacked Liouville states.  Pathway columns
    retain the exact order supplied by ``pathways``.
    """

    import Liouville_transport_general as general

    omega_values = np.asarray(omegas, dtype=float)
    pathway_list = tuple(pathways)

    def frame_fn(lam: Array) -> Array:
        return general.liouville_pathway_basis_matrix(
            model,
            parameter_map,
            np.asarray(lam, dtype=float),
            omega_values,
            pathway_list,
            stage=stage,
            tau2=tau2,
            eta=eta,
            quadrature_weight=quadrature_weight,
        )

    return frame_fn


def _validate_path(path: Array) -> Array:
    vertices = np.asarray(path, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("path must have shape (n_vertices, n_controls), n_vertices >= 2")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("path contains non-finite values")
    if np.any(np.linalg.norm(np.diff(vertices, axis=0), axis=1) == 0.0):
        raise ValueError("adjacent path vertices must be distinct")
    return vertices


def _collect_frames(
    path: Array,
    frames: Array | None,
    frame_fn: FrameFunction | None,
) -> Array:
    if (frames is None) == (frame_fn is None):
        raise ValueError("supply exactly one of frames or frame_fn")
    values = (
        np.asarray(frames, dtype=complex)
        if frames is not None
        else np.asarray([frame_fn(point) for point in path], dtype=complex)
    )
    if values.ndim != 3 or values.shape[0] != path.shape[0]:
        raise ValueError("frames must have shape (n_vertices, n_ambient, n_pathways)")
    if values.shape[1] < values.shape[2] or values.shape[2] == 0:
        raise ValueError("each frame must have 1 <= n_pathways <= n_ambient")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames contain non-finite values")
    return values


def _thin_polar_frame(frame: Array, rcond: float) -> tuple[Array, Array, int, float]:
    left, singular_values, right_h = np.linalg.svd(frame, full_matrices=False)
    threshold = rcond * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    if rank < frame.shape[1]:
        raise np.linalg.LinAlgError(
            f"pathway frame is rank deficient: rank={rank}, columns={frame.shape[1]}"
        )
    condition = float(singular_values[0] / singular_values[-1])
    return left @ right_h, singular_values, rank, condition


def _unitary_polar(matrix: Array) -> Array:
    left, _singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


def parallel_transport_pathway_frames(
    path: Array,
    *,
    frames: Array | None = None,
    frame_fn: FrameFunction | None = None,
    rcond: float = 1.0e-10,
) -> PathwayFrameTransportResult:
    """Transport an ordered Liouville pathway frame along a parameter path.

    The unitary polar overlap gives an isometric discrete connection.  Raw
    overlap projections are retained separately to quantify norm loss caused
    by finite subspace mismatch.
    """

    if not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie strictly between zero and one")
    vertices = _validate_path(path)
    raw = _collect_frames(vertices, frames, frame_fn)
    orthonormal = []
    frame_singular_values = []
    frame_ranks = []
    frame_conditions = []
    for frame in raw:
        q_frame, singular_values, rank, condition = _thin_polar_frame(frame, rcond)
        orthonormal.append(q_frame)
        frame_singular_values.append(singular_values)
        frame_ranks.append(rank)
        frame_conditions.append(condition)
    orthonormal_array = np.asarray(orthonormal)
    n_steps = vertices.shape[0] - 1
    n_pathways = raw.shape[2]
    identity = np.eye(n_pathways, dtype=complex)

    local_transport = []
    local_projection = []
    cumulative_transport = [identity]
    cumulative_projection = [identity]
    subspace_singular_values = []
    principal_angles = []
    overlap_ranks = []
    overlap_conditions = []
    local_unitarity_errors = []
    cumulative_gram_errors = [0.0]
    projection_gram_errors = [0.0]
    alignment_residuals = []
    aligned_frames = [orthonormal_array[0]]

    for index in range(n_steps):
        overlap = orthonormal_array[index + 1].conj().T @ orthonormal_array[index]
        singular_values = np.linalg.svd(overlap, compute_uv=False)
        clipped = np.clip(singular_values, 0.0, 1.0)
        threshold = rcond * max(singular_values[0], np.finfo(float).tiny)
        rank = int(np.sum(singular_values > threshold))
        if rank < n_pathways:
            raise np.linalg.LinAlgError(
                "adjacent pathway subspaces have a rank-deficient overlap: "
                f"step={index}, rank={rank}, columns={n_pathways}"
            )
        transport = _unitary_polar(overlap)
        cumulative = transport @ cumulative_transport[-1]
        projected = overlap @ cumulative_projection[-1]
        aligned = orthonormal_array[index + 1] @ cumulative

        local_transport.append(transport)
        local_projection.append(overlap)
        cumulative_transport.append(cumulative)
        cumulative_projection.append(projected)
        subspace_singular_values.append(singular_values)
        principal_angles.append(np.arccos(clipped))
        overlap_ranks.append(rank)
        overlap_conditions.append(float(singular_values[0] / singular_values[-1]))
        local_unitarity_errors.append(
            float(np.linalg.norm(transport.conj().T @ transport - identity, ord="fro"))
        )
        cumulative_gram_errors.append(
            float(np.linalg.norm(cumulative.conj().T @ cumulative - identity, ord="fro"))
        )
        projection_gram_errors.append(
            float(np.linalg.norm(projected.conj().T @ projected - identity, ord="fro"))
        )
        alignment_residuals.append(
            float(np.linalg.norm(aligned - aligned_frames[-1], ord="fro"))
        )
        aligned_frames.append(aligned)

    return PathwayFrameTransportResult(
        path=vertices,
        raw_frames=raw,
        orthonormal_frames=orthonormal_array,
        aligned_frames=np.asarray(aligned_frames),
        local_transport_maps=np.asarray(local_transport),
        cumulative_transport_maps=np.asarray(cumulative_transport),
        local_projection_maps=np.asarray(local_projection),
        cumulative_projection_maps=np.asarray(cumulative_projection),
        frame_singular_values=np.asarray(frame_singular_values),
        frame_ranks=np.asarray(frame_ranks, dtype=int),
        frame_condition_numbers=np.asarray(frame_conditions),
        subspace_singular_values=np.asarray(subspace_singular_values),
        principal_angles=np.asarray(principal_angles),
        overlap_ranks=np.asarray(overlap_ranks, dtype=int),
        overlap_condition_numbers=np.asarray(overlap_conditions),
        local_unitarity_errors=np.asarray(local_unitarity_errors),
        cumulative_gram_errors=np.asarray(cumulative_gram_errors),
        projection_gram_errors=np.asarray(projection_gram_errors),
        gauge_alignment_residuals=np.asarray(alignment_residuals),
    )


def refine_piecewise_linear_path(path: Array, factor: int) -> Array:
    """Subdivide every path segment by an integer factor."""

    vertices = _validate_path(path)
    if int(factor) != factor or factor < 1:
        raise ValueError("factor must be a positive integer")
    factor = int(factor)
    refined = [vertices[0]]
    for start, stop in zip(vertices[:-1], vertices[1:], strict=True):
        refined.extend(start + (step / factor) * (stop - start) for step in range(1, factor + 1))
    return np.asarray(refined)


def transport_with_refinement(
    path: Array,
    frame_fn: FrameFunction,
    *,
    refinement_factors: Sequence[int] = (1, 2, 4),
    rcond: float = 1.0e-10,
) -> PathwayFrameRefinementResult:
    """Solve on successively refined paths and report final-map convergence."""

    factors = tuple(int(value) for value in refinement_factors)
    if not factors or any(value < 1 for value in factors):
        raise ValueError("refinement_factors must contain positive integers")
    if any(second <= first for first, second in zip(factors[:-1], factors[1:])):
        raise ValueError("refinement_factors must be strictly increasing")
    results = tuple(
        parallel_transport_pathway_frames(
            refine_piecewise_linear_path(path, factor), frame_fn=frame_fn, rcond=rcond
        )
        for factor in factors
    )
    map_errors = []
    ambient_errors = []
    for coarse, fine in zip(results[:-1], results[1:], strict=True):
        map_errors.append(
            np.linalg.norm(fine.final_transport - coarse.final_transport, ord="fro")
        )
        ambient_errors.append(
            np.linalg.norm(fine.aligned_frames[-1] - coarse.aligned_frames[-1], ord="fro")
        )
    return PathwayFrameRefinementResult(
        refinement_factors=factors,
        results=results,
        successive_transport_errors=np.asarray(map_errors),
        successive_ambient_frame_errors=np.asarray(ambient_errors),
    )


__all__ = [
    "PathwayFrameRefinementResult",
    "PathwayFrameTransportResult",
    "build_liouville_pathway_frame_function",
    "parallel_transport_pathway_frames",
    "refine_piecewise_linear_path",
    "transport_with_refinement",
]
