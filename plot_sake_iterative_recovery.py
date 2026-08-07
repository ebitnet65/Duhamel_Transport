#!/usr/bin/env python3
"""Plot parameter trajectory and direct/surrogate residuals for inverse SAKE."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.result_dir / "iterations.csv").open()))
    summary = json.loads((args.result_dir / "summary.json").read_text())
    references = np.array([[float(row["reference_J"]), float(row["reference_k"])] for row in rows])
    final = np.asarray(summary["recovered"], dtype=float)
    trajectory = np.vstack([references, final]) * 1000.0
    target = np.asarray(summary["target"], dtype=float) * 1000.0
    iterations = np.arange(1, len(rows) + 1)
    predicted = np.array([float(row["predicted_trial_error"]) for row in rows])
    direct = np.array([float(row["direct_trial_error"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], "o-", label="accepted chart centers")
    axes[0].plot(target[0], target[1], "*", markersize=14, label="target")
    for index, point in enumerate(trajectory):
        axes[0].annotate(str(index), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel(r"$J$ / meV")
    axes[0].set_ylabel(r"$\kappa$ / meV")
    axes[0].set_title("Trust-region trajectory")
    axes[0].legend()

    axes[1].semilogy(iterations, predicted, "s--", label="SAKE prediction")
    axes[1].semilogy(iterations, direct, "o-", label="direct validation")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative pathway-state error")
    axes[1].set_title("Inverse convergence")
    axes[1].legend()
    axes[1].set_xticks(iterations)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.result_dir / f"iterative_recovery.{suffix}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
