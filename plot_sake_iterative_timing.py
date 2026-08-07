#!/usr/bin/env python3
"""Plot direct-response and per-step wall-clock costs for iterative SAKE recovery."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, list[dict[str, str]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with (path / "iterations.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path) for path in args.runs]
    labels = ["T1: (-0.050, 0.006)", "T2: (0.040, 0.004)", "T3: (-0.030, 0.010)"]
    colors = {
        "optimizer_seconds": "#b8d8eb",
        "direct_validation_seconds": "#72a9cf",
        "transport_projection_seconds": "#f2c879",
        "chart_rebuild_seconds": "#ed9b66",
    }
    names = {
        "optimizer_seconds": "surrogate optimization",
        "direct_validation_seconds": "direct validation",
        "transport_projection_seconds": "transport projection",
        "chart_rebuild_seconds": "local-chart rebuild",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, constrained_layout=True)
    summary_rows = []
    for run_index, (axis, (summary, rows), label) in enumerate(zip(axes, runs, labels, strict=True)):
        direct = float(summary["target_direct_response_seconds"])
        initial_chart = float(summary["initial_chart_seconds"])
        x = np.arange(len(rows) + 2)
        axis.bar(0, direct, color="#d9d9d9", edgecolor="0.35", linewidth=0.5, label="one direct response")
        axis.bar(1, initial_chart, color="#f6d7bd", edgecolor="0.35", linewidth=0.5, label="initial chart")
        bottom = np.zeros(len(rows))
        for key in ("optimizer_seconds", "direct_validation_seconds", "transport_projection_seconds", "chart_rebuild_seconds"):
            values = np.array([float(row[key]) for row in rows])
            axis.bar(np.arange(2, len(rows) + 2), values, bottom=bottom, color=colors[key], label=names[key])
            bottom += values
        axis.axhline(direct, color="0.25", linestyle="--", linewidth=0.8)
        axis.set_title(label, fontsize=8)
        axis.set_xticks(x, ["Direct", "Chart 0"] + [str(i + 1) for i in range(len(rows))], rotation=45, ha="right")
        axis.set_xlabel("response or accepted step")
        axis.tick_params(direction="in", top=True, right=True, labelsize=6.5)
        mean_step = float(np.mean([float(row["iteration_total_seconds"]) for row in rows]))
        summary_rows.append({
            "target_J_over_omega0": float(summary["target"][0]) / 2.0,
            "target_kappa_over_omega0": float(summary["target"][1]) / 2.0,
            "direct_response_seconds": direct,
            "initial_chart_seconds": initial_chart,
            "accepted_steps": int(summary["accepted_steps"]),
            "mean_step_seconds": mean_step,
            "iterative_total_seconds": float(summary["iterative_total_seconds"]),
            "optimizer_fraction": float(summary["optimizer_total_seconds"]) / float(summary["iterative_total_seconds"]),
            "direct_validation_fraction": float(summary["direct_validation_total_seconds"]) / float(summary["iterative_total_seconds"]),
            "transport_projection_fraction": float(summary["transport_projection_total_seconds"]) / float(summary["iterative_total_seconds"]),
            "chart_rebuild_fraction": float(summary["chart_rebuild_total_seconds"]) / float(summary["iterative_total_seconds"]),
        })
    axes[0].set_ylabel("wall time (s)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, frameon=False, fontsize=6.5)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"iterative_wall_time_breakdown.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    with (args.output_dir / "iterative_wall_time_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
