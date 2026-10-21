#!/usr/bin/env python3
"""Generate the publication figure for Fourier-window parameter sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fourier_window_publication import (
    CONTROL_LABELS,
    configure_matplotlib,
    load_checked_benchmark,
    save_figure,
    write_control_csv,
)


def generate(input_dir: Path, output_dir: Path):
    payload, _arrays, strengths, diagnostics, endpoints, truth, _reference = (
        load_checked_benchmark(input_dir)
    )
    x = 1.0e3 * strengths
    normalized = endpoints / truth
    residual = np.asarray([item["relative_orthogonal_residual"] for item in diagnostics])
    holonomy = np.asarray([
        route["pathway_route_comparison"]["holonomy_frobenius_distance_from_identity"]
        for route in payload["routes"]
    ])
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "^")
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), constrained_layout=True)

    for index, (label, color, marker) in enumerate(zip(CONTROL_LABELS, colors, markers)):
        axes[0].plot(x, normalized[:, index], marker=marker, color=color, label=label)
        axes[1].plot(
            x, 100.0 * (endpoints[:, index] - truth[index]) / truth[index],
            marker=marker, color=color, label=label,
        )
    axes[0].axhline(1.0, color="0.35", lw=0.8, ls=":")
    axes[0].set_ylabel(r"recovered control / truth")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="0.35", lw=0.8, ls=":")
    axes[1].set_ylabel("control bias (%)")

    axes[2].semilogy(x, np.maximum(residual, 1e-12), "o-", color="#CC79A7", label=r"$\|r_\perp\|/\|y\|$")
    twin = axes[2].twinx()
    twin.plot(x, holonomy, "D-", color="#56B4E9", label="route holonomy")
    axes[2].set_ylabel("orthogonal residual")
    twin.set_ylabel("holonomy distance", color="#0072B2")
    twin.tick_params(axis="y", colors="#0072B2")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="lower right")

    for panel, ax in zip("abc", axes, strict=True):
        ax.axvspan(0.0, 1.0, color="#009E73", alpha=0.10, lw=0)
        ax.axvline(1.0, color="#009E73", lw=0.9, ls="--")
        ax.set_xlabel(r"apodization strength ($10^{-3}$)")
        ax.text(0.02, 0.97, f"({panel})", transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(alpha=0.18, lw=0.5)
    save_figure(fig, output_dir, "figure_fourier_window_sensitivity")
    plt.close(fig)
    # Include route holonomy in the audit CSV without altering the source archive.
    enriched = []
    for item, value in zip(diagnostics, holonomy, strict=True):
        record = dict(item)
        record["route_holonomy_distance"] = float(value)
        enriched.append(record)
    write_control_csv(
        output_dir / "figure_fourier_window_sensitivity.csv",
        strengths, enriched, endpoints, truth,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("fourier_artifact_robustness_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("publication_artifacts/fourier_window"))
    args = parser.parse_args(argv)
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
