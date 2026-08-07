#!/usr/bin/env python3
"""Create publication plots and a summary table for multiple SAKE recoveries."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def load_run(path: Path) -> dict:
    rows = list(csv.DictReader((path / "iterations.csv").open()))
    summary = json.loads((path / "summary.json").read_text())
    reference = np.array([[float(row["reference_J"]), float(row["reference_k"])] for row in rows])
    recovered = np.asarray(summary["recovered"], dtype=float)
    return {"path": path, "rows": rows, "summary": summary, "trajectory": np.vstack([reference, recovered])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omega0", type=float, default=2.0, help="Mean transition energy in eV.")
    parser.add_argument(
        "--normalize-controls",
        action="store_true",
        help="Plot J/omega0 and kappa/omega0 instead of meV.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path) for path in args.result_dirs]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    summary_rows = []

    for index, run in enumerate(runs):
        color = colors[index]
        target_physical = np.asarray(run["summary"]["target"])
        if args.normalize_controls:
            target = target_physical / args.omega0
            trajectory = run["trajectory"] / args.omega0
            label = rf"T{index + 1}: ({target[0]:.3f}, {target[1]:.3f})"
        else:
            target = target_physical * 1000.0
            trajectory = run["trajectory"] * 1000.0
            label = rf"T{index + 1}: ({target[0]:.0f}, {target[1]:.0f}) meV"
        axes[0].plot(
            trajectory[:, 0], trajectory[:, 1], "o-", color=color,
            markersize=3.8, label=label,
        )
        axes[0].plot(target[0], target[1], marker="*", color=color, markersize=9, linestyle="none")

        iteration = np.arange(1, len(run["rows"]) + 1)
        predicted = np.array([float(row["predicted_trial_error"]) for row in run["rows"]])
        direct = np.array([float(row["direct_trial_error"]) for row in run["rows"]])
        axes[1].semilogy(iteration, direct, "o-", color=color, markersize=3.8)
        axes[1].semilogy(iteration, predicted, "s--", color=color, markersize=3.2, alpha=0.8)
        summary_rows.append({
            "target_J_meV": target[0],
            "target_k_meV": target[1],
            "recovered_J_meV": 1000.0 * run["summary"]["recovered"][0],
            "recovered_k_meV": 1000.0 * run["summary"]["recovered"][1],
            "parameter_error_meV": 1000.0 * run["summary"]["parameter_error_eV"],
            "direct_validations": run["summary"]["direct_trial_validations"],
            "final_state_error": run["summary"]["final_state_relative_error"],
            "transport_error": run["summary"]["recovered_vs_target_transport_error"],
            "off_diagonal_fraction": run["summary"]["accumulated_off_diagonal_fraction"],
        })

    if args.normalize_controls:
        axes[0].set_xlabel(r"$J/\omega_0$")
        axes[0].set_ylabel(r"$\kappa/\omega_0$")
        bound_value = 0.030 / args.omega0
        bound_label = rf"$\kappa_{{\max}}/\omega_0={bound_value:.3f}$"
    else:
        axes[0].set_xlabel(r"$J$ (meV)")
        axes[0].set_ylabel(r"$\kappa$ (meV)")
        bound_value = 30.0
        bound_label = r"$\kappa_{\max}=30$ meV"
    axes[0].set_title("(a) Accepted trust-region trajectory", loc="left")
    axes[0].axhline(bound_value, color="0.65", linewidth=0.8, linestyle=":")
    trajectory_handles, trajectory_labels = axes[0].get_legend_handles_labels()
    trajectory_handles.append(Line2D([0], [0], color="0.55", linewidth=0.8, linestyle=":"))
    trajectory_labels.append(bound_label)
    axes[0].legend(trajectory_handles, trajectory_labels, frameon=False, loc="best")

    axes[1].set_xlabel("Accepted iteration")
    axes[1].set_ylabel("Relative pathway-state error")
    axes[1].set_title("(b) Pathway-state residual during recovery", loc="left")
    style_legend = [
        Line2D([0], [0], marker="o", color="0.2", label="Direct validation", markersize=3.8),
        Line2D([0], [0], marker="s", color="0.2", linestyle="--", label="SAKE prediction", markersize=3.2),
    ]
    axes[1].legend(handles=style_legend, frameon=False, loc="best")
    axes[1].set_xticks(np.arange(1, max(len(run["rows"]) for run in runs) + 1))
    for axis in axes:
        axis.grid(True, which="major", color="0.88", linewidth=0.6)
        axis.tick_params(direction="in", top=True, right=True)

    figure_stem = "multi_target_recovery_normalized" if args.normalize_controls else "multi_target_recovery"
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"{figure_stem}.{suffix}", dpi=400)
    plt.close(fig)

    with (args.output_dir / "multi_target_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"$J_\mathrm{target}$ & $\kappa_\mathrm{target}$ & $J_\mathrm{fit}$ & $\kappa_\mathrm{fit}$ & $\|\Delta\theta\|$ & Direct checks \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['target_J_meV']:.0f} & {row['target_k_meV']:.0f} & "
            f"{row['recovered_J_meV']:.3f} & {row['recovered_k_meV']:.3f} & "
            f"{row['parameter_error_meV']:.3g} & {row['direct_validations']} " + r"\\"
        )
    lines.extend([r"\end{tabular}", ""])
    (args.output_dir / "multi_target_summary.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
