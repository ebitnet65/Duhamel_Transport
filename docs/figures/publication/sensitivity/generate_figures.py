#!/usr/bin/env python3
"""Generate publication figures from the sensitivity benchmark.

The atlas directly reevaluates the repository forward model and autodiff
Jacobian on a configurable dense frequency grid. Waiting-time diagnostics use
the stored benchmark information record. No interpolation or benchmark edits
are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PARAMS = (r"$J_{12}$", r"$k_{12}$", r"$\gamma_\phi$")
PARAM_UNITS = ("energy", "rate", "rate")


def _read(source: Path):
    npz = np.load(source / "spectral_sensitivity_benchmark.npz")
    with (source / "spectral_sensitivity_benchmark.json").open() as f:
        meta = json.load(f)
    return npz, meta


def _map(ax, z, grid, *, cmap, norm=None, title="", cbar_label="", xlabel=True,
         ylabel=True):
    # One rendered square per recorded sample: no synthetic frequency detail.
    step = float(np.median(np.diff(grid))) if len(grid) > 1 else 1.0
    edges = [grid[0] - step/2, grid[-1] + step/2]
    im = ax.imshow(z, origin="lower", interpolation="nearest", cmap=cmap, norm=norm,
                   extent=[edges[0], edges[1], edges[0], edges[1]], aspect="equal")
    ax.set_title(title, loc="left", fontsize=10, pad=4)
    tick_idx = np.unique(np.linspace(0, len(grid) - 1, min(5, len(grid))).round().astype(int))
    ax.set_xticks(grid[tick_idx])
    ax.set_yticks(grid[tick_idx])
    ax.tick_params(labelsize=7, length=2)
    # Observable arrays are [omega3, omega1], so image columns map to omega1.
    ax.set_xlabel(r"$\omega_1$ (model energy units)" if xlabel else "", fontsize=8)
    ax.set_ylabel(r"$\omega_3$ (model energy units)" if ylabel else "", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.035)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(cbar_label, fontsize=7)
    return im


def dense_evaluation(npz, meta, out: Path, nfreq: int, benchmark_dir: Path):
    """Evaluate baseline spectra and derivatives on a true dense model grid."""
    import jax
    import jax.numpy as jnp
    from dimer_spectral_observable import DimerSpectrumConfig, build_dimer_spectral_observable

    q = np.asarray(npz["q0"], float)
    waits = np.asarray(npz["waiting_times"], float)
    grid = np.linspace(float(np.min(npz["omega_grid"])), float(np.max(npz["omega_grid"])), nfreq)
    eta = float(meta["grid"]["eta"])
    cfg = DimerSpectrumConfig(grid, grid, eta=eta)
    base = build_dimer_spectral_observable(cfg)
    rp = build_dimer_spectral_observable(DimerSpectrumConfig(grid, grid, eta=eta, sectors=("RP",)))
    nrp = build_dimer_spectral_observable(DimerSpectrumConfig(grid, grid, eta=eta, sectors=("NRP",)))
    ti = int(np.argmin(np.abs(waits - 1.0)))
    t = float(waits[ti])
    qj = jnp.asarray(q)
    spectra = np.asarray(base(qj, t))
    spectra_rp = np.asarray(rp(qj, t))
    spectra_nrp = np.asarray(nrp(qj, t))
    jac = np.moveaxis(np.asarray(jax.jacfwd(lambda x: base(x, t))(qj)), -1, 0)
    scales = np.asarray(npz["parameter_scales"])
    sigma = float(meta["noise"]["independent_complex_gaussian_cartesian_sigma"])
    jflat = jac.reshape(3, -1).T
    jreal = np.vstack((jflat.real, jflat.imag)) / sigma
    info = jreal.T @ jreal
    g = np.diag(1.0 / scales**2)
    values, vectors = eigh(info, g)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    np.savez_compressed(out / "dense_sensitivity_plot_data.npz", omega_grid=grid,
                        waiting_time=t, q0=q, spectra=spectra, spectra_RP=spectra_rp,
                        spectra_NRP=spectra_nrp, jacobian=jac,
                        generalized_eigenvalues=values, generalized_eigenvectors=vectors,
                        parameter_scales=scales, sigma=sigma)
    source_npz = benchmark_dir / "spectral_sensitivity_benchmark.npz"
    source_json = benchmark_dir / "spectral_sensitivity_benchmark.json"
    hashes = {}
    for path in (source_npz, source_json):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {"model": "dimer_spectral_observable.build_dimer_spectral_observable",
                  "parameters": {"J12": float(q[0]), "k12": float(q[1]), "gamma_phi": float(q[2])},
                  "waiting_time": t, "frequency_grid": {"min": float(grid[0]), "max": float(grid[-1]),
                                       "points_per_axis": int(nfreq), "direct_forward_evaluation": True},
                  "eta": eta, "noise_sigma_for_information_modes": sigma,
                  "parameter_reference_scales": scales.tolist(),
                  "jacobian": "JAX jacfwd of complex RP+NRP observable; parameter axis moved to leading position",
                  "source_benchmark_sha256": hashes,
                  "note": "Dense model-evaluated visualization grid; not an experimental frequency sampling claim."}
    (out / "dense_sensitivity_plot_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return grid, t, spectra, spectra_rp, spectra_nrp, jac, scales, values, vectors


def atlas(dense, out: Path):
    grid, t, s, s_rp, s_nrp, jac, scales, vals, vectors = dense
    # Dense-grid modes are recomputed from this figure's model-evaluated Jacobian.
    mode = vectors[:, -1]
    mode_map = np.einsum("p,pij->ij", mode, jac)

    fig, axs = plt.subplots(3, 4, figsize=(14.2, 9.3), constrained_layout=True)
    fig.suptitle(f"Spectral sensitivity atlas\nDimer control point; $t_2={t:g}$; "
                 f"{len(grid)} × {len(grid)} dense model-evaluated pixels; complex response retained",
                 fontsize=15, fontweight="bold")
    spectral_maps = [(s_rp.real, "RP · Re", "signal"),
                     (s_nrp.real, "NRP · Re", "signal"),
                     (s.real, "RP + NRP · Re", "signal"),
                     (s.imag, "RP + NRP · Im", "signal")]
    lim = max(float(np.max(np.abs(z))) for z, _, _ in spectral_maps)
    for ax, (z, title, unit) in zip(axs[0], spectral_maps):
        _map(ax, z, grid, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
             title=title, cbar_label=unit, xlabel=True, ylabel=True)

    deriv = [scales[i] * jac[i] for i in range(3)]
    dlim = max(float(np.max(np.abs(z.real))) for z in deriv + [mode_map])
    for i, ax in enumerate(axs[1, :3]):
        _map(ax, deriv[i].real, grid, cmap="RdBu_r",
             norm=TwoSlopeNorm(vmin=-dlim, vcenter=0, vmax=dlim),
             title=rf"Coordinate perturbation · {PARAMS[i]}", cbar_label=r"Re $\Delta S$ / unit normalized step")
    mode_lim_r = float(np.max(np.abs(mode_map.real))) or 1.0
    _map(axs[1, 3], mode_map.real, grid, cmap="RdBu_r",
         norm=TwoSlopeNorm(vmin=-mode_lim_r, vcenter=0, vmax=mode_lim_r),
         title=rf"Weakest generalized mode · $\chi={vals[-1]:.2g}$", cbar_label=r"Re $\Delta S$ / unit $G$-length")

    dlim_i = max(float(np.max(np.abs(z.imag))) for z in deriv + [mode_map])
    for i, ax in enumerate(axs[2, :3]):
        _map(ax, deriv[i].imag, grid, cmap="RdBu_r",
             norm=TwoSlopeNorm(vmin=-dlim_i, vcenter=0, vmax=dlim_i),
             title=rf"Coordinate perturbation · {PARAMS[i]}", cbar_label=r"Im $\Delta S$ / unit normalized step")
    mode_lim_i = float(np.max(np.abs(mode_map.imag))) or 1.0
    _map(axs[2, 3], mode_map.imag, grid, cmap="RdBu_r",
         norm=TwoSlopeNorm(vmin=-mode_lim_i, vcenter=0, vmax=mode_lim_i),
         title="Weakest generalized mode", cbar_label=r"Im $\Delta S$ / unit $G$-length")
    fig.text(.5, -.012,
             rf"Coordinate steps use $s_i\,\partial S/\partial\lambda_i$, with scales "
             rf"$s=({scales[0]:g}\ {PARAM_UNITS[0]}, {scales[1]:g}\ {PARAM_UNITS[1]}, "
             rf"{scales[2]:g}\ {PARAM_UNITS[2]})$. Frequency pixels are shown as recorded; "
             "the dense map is a forward-model evaluation, not an experimental sampling resolution.",
             ha="center", fontsize=8)
    fig.savefig(out / "figure_spectral_sensitivity_atlas.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_spectral_sensitivity_atlas.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def observability(npz, meta, out: Path):
    waits = np.asarray(npz["waiting_times"])
    scales = np.asarray(npz["parameter_scales"])
    g = np.diag(1.0 / scales**2)
    diags = [meta["baseline_by_waiting_time"][str(float(t))] for t in waits]
    singular = np.array([d["singular_values"] for d in diags])
    eigvals = np.array([d["generalized_eigenvalues"] for d in diags])
    vecs = [np.asarray(d["generalized_eigenvectors"]) for d in diags]
    cors = np.array([d["parameter_correlation"] for d in diags])
    # Match each next-time basis to the prior basis by maximum absolute G-overlap.
    tracked = [vecs[0].copy()]
    for k in range(1, len(waits)):
        overlap = np.abs(tracked[-1].T @ g @ vecs[k])
        rows, cols = linear_sum_assignment(-overlap)
        perm = cols[np.argsort(rows)]
        tracked.append(vecs[k][:, perm])
        eigvals[k] = eigvals[k, perm]
    tracks = np.array(tracked)

    fig = plt.figure(figsize=(14, 8.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    for j, label in enumerate(("Mode 1", "Mode 2", "Mode 3")):
        ax.plot(waits, eigvals[:, j], marker="o", lw=2, label=label)
    ax.set_yscale("log"); ax.set_xlabel(r"Waiting time $t_2$"); ax.set_ylabel(r"Generalized sensitivity $\chi$")
    ax.set_title("A · Generalized modes (overlap tracked)", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for j, label in enumerate(("1", "2", "3")):
        ax.plot(waits, singular[:, j], marker="o", lw=2, label=f"Singular value {label}")
    ax.set_yscale("log"); ax.set_xlabel(r"Waiting time $t_2$"); ax.set_ylabel("Whitened Jacobian singular value")
    ax.set_title("B · Local spectral information", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    cond = [d["condition"] for d in diags]
    ax.plot(waits, cond, marker="o", lw=2.2, color="#7b3294")
    ax.set_xlabel(r"Waiting time $t_2$"); ax.set_ylabel("Jacobian condition number")
    ax.set_title("C · Conditioning", loc="left", fontweight="bold")
    ax.grid(True, alpha=.25)

    for k, ax in enumerate([fig.add_subplot(gs[1, i]) for i in range(3)]):
        im = ax.imshow(cors[k], vmin=-1, vmax=1, cmap="RdBu_r", origin="upper", interpolation="nearest")
        ax.set_xticks(range(3), PARAMS); ax.set_yticks(range(3), PARAMS)
        ax.tick_params(labelsize=9)
        ax.set_title(rf"D{k+1} · Parameter correlations, $t_2={waits[k]:g}$", loc="left", fontweight="bold")
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{cors[k,i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, ticks=[-1, 0, 1], label="Correlation")

    fig.suptitle("Waiting-time observability of the three-parameter dimer", fontsize=16, fontweight="bold")
    fig.text(.5, -.012,
             "Information uses equal independent Cartesian noise weights at each sampled complex pixel. "
             "The physical metric is fixed across waiting times: G = diag(1/sᵢ²), with the benchmark reference scales. "
             "Information diagnostics use the recorded coarse 3 × 3 benchmark grid; eigenvectors are matched by maximum G-overlap.",
             ha="center", fontsize=8)
    fig.savefig(out / "figure_waiting_time_observability.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_waiting_time_observability.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Include derived mode tracking and plotting values, so the figure can be audited.
    payload = {"waiting_times": waits.tolist(), "singular_values": singular.tolist(),
               "generalized_eigenvalues_tracked": eigvals.tolist(),
               "tracked_generalized_eigenvectors_columns": [v.tolist() for v in tracks],
               "parameter_correlations": cors.tolist(), "condition_numbers": cond,
               "physical_metric": g.tolist(), "tracking": "Hungarian assignment maximizing absolute G-inner-product overlaps"}
    (out / "waiting_time_observability_plot_data.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("docs/benchmarks/spectral_sensitivity_model_discrimination"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/publication/sensitivity"))
    parser.add_argument("--nfreq-dense", type=int, default=21,
                        help="points per frequency axis for direct forward-model atlas evaluation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz, meta = _read(args.benchmark_dir)
    if args.nfreq_dense < 5:
        parser.error("--nfreq-dense must be at least 5")
    dense = dense_evaluation(npz, meta, args.output_dir, args.nfreq_dense, args.benchmark_dir)
    atlas(dense, args.output_dir)
    observability(npz, meta, args.output_dir)
    print(f"Wrote publication figures to {args.output_dir}")


if __name__ == "__main__":
    main()
