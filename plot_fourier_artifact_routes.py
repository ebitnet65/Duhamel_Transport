#!/usr/bin/env python3
"""Headless exploratory figures for Fourier-artifact endpoint and route drift."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sake-dt")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = (r"$J_{12}$", r"$k_{12}$", r"$\gamma_\phi$")


def plot_artifact_routes(json_path: Path, npz_path: Path, output_dir: Path):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    diagnostics = payload["discrepancy"]["diagnostics"]
    strengths = np.asarray(payload["discrepancy"]["artifact_strengths"])
    truth, reference = data["truth"], data["reference"]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(strengths.min(), strengths.max() or 1.0)

    fig = plt.figure(figsize=(8.2, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*reference, marker="s", s=65, color="black", label="reference $p$")
    ax.scatter(*truth, marker="*", s=180, color="crimson", label="truth $q_{true}$")
    for index, item in enumerate(diagnostics):
        color = cmap(norm(item["artifact_strength"]))
        straight, geodesic = data[f"straight_path_{index}"], data[f"geodesic_path_{index}"]
        ax.plot(*straight.T, ls="--", lw=0.9, color=color, alpha=0.45)
        ax.plot(*geodesic.T, lw=2.0, color=color)
        endpoint = data[f"endpoint_{index}"]
        marker = "o" if item["reliable"] else "x"
        size = 45 + 1200 * min(float(item["physical_uncertainty_rms"]), 0.05)
        scatter_options = {"edgecolor": "black"} if marker == "o" else {}
        ax.scatter(*endpoint, marker=marker, s=size, color=color, **scatter_options)
    ax.set_xlabel(LABELS[0]); ax.set_ylabel(LABELS[1]); ax.set_zlabel(LABELS[2])
    ax.set_title("Fourier-artifact endpoint drift and physical routes")
    ax.legend(loc="upper left", fontsize=8)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.10, shrink=0.75)
    colorbar.set_label("exponential apodization strength")
    fig.savefig(output_dir / "fourier_artifact_routes_3d.png", dpi=180)
    fig.savefig(output_dir / "fourier_artifact_routes_3d.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for ax, (x, y) in zip(axes, ((0, 1), (0, 2), (1, 2)), strict=True):
        ax.scatter(reference[x], reference[y], marker="s", color="black", s=35)
        ax.scatter(truth[x], truth[y], marker="*", color="crimson", s=100)
        for index, item in enumerate(diagnostics):
            color = cmap(norm(item["artifact_strength"]))
            straight, geodesic = data[f"straight_path_{index}"], data[f"geodesic_path_{index}"]
            ax.plot(straight[:, x], straight[:, y], "--", lw=0.8, color=color, alpha=0.4)
            ax.plot(geodesic[:, x], geodesic[:, y], lw=1.6, color=color)
            covariance = data[f"covariance_{index}"][np.ix_([x, y], [x, y])]
            values, vectors = np.linalg.eigh(covariance)
            theta = np.linspace(0, 2*np.pi, 80)
            ellipse = vectors @ (np.sqrt(np.maximum(values, 0))[:, None] * np.vstack((np.cos(theta), np.sin(theta))))
            endpoint = data[f"endpoint_{index}"][[x, y]]
            ax.plot(endpoint[0] + ellipse[0], endpoint[1] + ellipse[1], color=color, lw=0.8, alpha=0.7)
            ax.scatter(*endpoint, marker="o" if item["reliable"] else "x", color=color, s=22)
        ax.set_xlabel(LABELS[x]); ax.set_ylabel(LABELS[y]); ax.grid(alpha=0.2)
    window = payload["discrepancy"]["uncertainty_window"]
    fig.suptitle(f"Control-plane projections; reliable artifact window: {window}")
    fig.savefig(output_dir / "fourier_artifact_route_projections.png", dpi=180)
    fig.savefig(output_dir / "fourier_artifact_route_projections.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_artifact_routes(args.json, args.npz, args.output_dir)


if __name__ == "__main__": main()
