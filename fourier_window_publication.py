"""Shared, import-safe helpers for Fourier-window publication figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_STRENGTHS = np.asarray([0.0, 0.0005, 0.001, 0.002, 0.005, 0.01])
CONTROL_LABELS = (r"$J_{12}$", r"$k_{12}$", r"$\gamma_\phi$")
CONTROL_NAMES = ("J_12", "k_12", "gamma_phi")


def load_checked_benchmark(input_dir: Path):
    """Load and validate the exact benchmark used for the paper figures."""

    payload = json.loads((input_dir / "fourier_artifact_routes.json").read_text())
    arrays = np.load(input_dir / "fourier_artifact_routes.npz")
    strengths = np.asarray(payload["discrepancy"]["artifact_strengths"], dtype=float)
    if payload["artifact_family"] != "symmetric_exponential_apodization":
        raise ValueError("publication figure requires symmetric exponential apodization")
    if not np.array_equal(strengths, EXPECTED_STRENGTHS):
        raise ValueError(f"unexpected artifact grid: {strengths}")
    if len(payload["frequency_grid"]) != 3 or float(payload["t2"]) != 0.8:
        raise ValueError("publication figure requires the archived 3x3, t2=0.8 benchmark")
    diagnostics = payload["discrepancy"]["diagnostics"]
    endpoints = np.vstack([item["endpoint"] for item in diagnostics])
    truth = np.asarray(arrays["truth"], dtype=float)
    reference = np.asarray(arrays["reference"], dtype=float)
    if not np.all(np.isfinite(endpoints)):
        raise ValueError("non-finite endpoint in benchmark")
    return payload, arrays, strengths, diagnostics, endpoints, truth, reference


def write_control_csv(path: Path, strengths, diagnostics, endpoints, truth):
    """Write compact numerical data sufficient to audit the sensitivity plot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "apodization_strength", *CONTROL_NAMES,
            *(f"{name}_over_truth" for name in CONTROL_NAMES),
            "relative_orthogonal_residual", "route_holonomy_distance", "reliable",
        ])
        for index, (strength, item, endpoint) in enumerate(
            zip(strengths, diagnostics, endpoints, strict=True)
        ):
            writer.writerow([
                float(strength), *map(float, endpoint), *map(float, endpoint / truth),
                float(item["relative_orthogonal_residual"]),
                float(item.get("route_holonomy_distance", np.nan)),
                bool(item["reliable"]),
            ])


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sake-dt-publication")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
