#!/usr/bin/env python3
"""Generate publication control-manifold projections for Fourier-window bias."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fourier_window_publication import (
    CONTROL_LABELS,
    configure_matplotlib,
    load_checked_benchmark,
    save_figure,
)


def generate(input_dir: Path, output_dir: Path):
    payload, arrays, strengths, diagnostics, endpoints, truth, reference = load_checked_benchmark(input_dir)
    plt = configure_matplotlib()
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(strengths.min(), strengths.max())
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    pairs = ((0, 1), (0, 2), (1, 2))
    for panel, (ax, (x_index, y_index)) in enumerate(zip(axes, pairs, strict=True)):
        ax.scatter(reference[x_index], reference[y_index], marker="s", s=35, color="black", zorder=5)
        ax.scatter(truth[x_index], truth[y_index], marker="*", s=95, color="#D81B60", zorder=6)
        for index, item in enumerate(diagnostics):
            color = cmap(norm(item["artifact_strength"]))
            straight = arrays[f"straight_path_{index}"]
            geodesic = arrays[f"geodesic_path_{index}"]
            ax.plot(straight[:, x_index], straight[:, y_index], ls="--", lw=0.75, color=color, alpha=0.55)
            ax.plot(geodesic[:, x_index], geodesic[:, y_index], lw=1.45, color=color)
            marker = "o" if item["reliable"] else "x"
            ax.scatter(
                endpoints[index, x_index], endpoints[index, y_index], marker=marker,
                s=25, color=color, edgecolors="black" if marker == "o" else None,
                linewidths=0.5 if marker == "o" else 1.1, zorder=7,
            )
        ax.set_xlabel(CONTROL_LABELS[x_index])
        ax.set_ylabel(CONTROL_LABELS[y_index])
        ax.grid(alpha=0.18, lw=0.5)
        ax.text(0.02, 0.97, f"({chr(97 + panel)})", transform=ax.transAxes, va="top", fontweight="bold")
    fig.legend(
        handles=[
            plt.Line2D([], [], marker="s", color="black", ls="", label="reference $p$"),
            plt.Line2D([], [], marker="*", color="#D81B60", ls="", markersize=9, label="truth $q_{true}$"),
            plt.Line2D([], [], color="0.35", ls="--", label="straight"),
            plt.Line2D([], [], color="0.35", label="geodesic"),
            plt.Line2D([], [], marker="o", color="0.35", ls="", label="accepted endpoint"),
            plt.Line2D([], [], marker="x", color="0.35", ls="", label="failed endpoint"),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.04),
        ncol=3, fontsize=6.8, columnspacing=1.2, handletextpad=0.5,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes, pad=0.015, aspect=25)
    colorbar.set_label("exponential apodization strength")
    save_figure(fig, output_dir, "figure_fourier_window_paths")
    plt.close(fig)

    # Archive the exact plotted path coordinates in a compact portable file.
    np.savez_compressed(
        output_dir / "figure_fourier_window_paths.npz",
        strengths=strengths,
        truth=truth,
        reference=reference,
        endpoints=endpoints,
        reliable=np.asarray([item["reliable"] for item in diagnostics], dtype=bool),
        **{f"straight_path_{i}": arrays[f"straight_path_{i}"] for i in range(len(strengths))},
        **{f"geodesic_path_{i}": arrays[f"geodesic_path_{i}"] for i in range(len(strengths))},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("fourier_artifact_robustness_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("publication_artifacts/fourier_window"))
    args = parser.parse_args(argv)
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
