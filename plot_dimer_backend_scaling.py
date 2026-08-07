#!/usr/bin/env python3
"""
plot_dimer_backend_scaling.py

Plot timing and accuracy scaling for dimer transport backends from benchmark
CSV files produced by `benchmark_dimer_transport_backends.py`.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv

import numpy as np

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")
import matplotlib.pyplot as plt


def read_rows(files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for file in files:
        with file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["backend"],
                    int(row["n_omega"]),
                    int(row.get("max_order", "2") or 2),
                    str(file.parent),
                )
                if key in seen:
                    continue
                seen.add(key)
                row["source"] = str(file.parent)
                rows.append(row)
    return rows


def write_combined_csv(rows: list[dict[str, str]], output_file: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def label_for(row: dict[str, str]) -> str:
    backend = row["backend"].replace("_", " ")
    max_order = row.get("max_order", "2") or "2"
    if row["backend"] == "duhamel_autodiff":
        return f"{backend}, max order {max_order}"
    return backend


def numeric(row: dict[str, str], key: str, fallback_key: str | None = None) -> float:
    value = row.get(key, "")
    if value != "":
        return float(value)
    if fallback_key is not None:
        return float(row[fallback_key])
    raise KeyError(key)


def plot_timing(rows: list[dict[str, str]], output_file: Path) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(label_for(row), []).append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for label, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: int(item["n_omega"]))
        n = np.asarray([int(item["n_omega"]) for item in group], dtype=float)
        seconds = np.asarray([float(item["backend_seconds"]) for item in group], dtype=float)
        ax.plot(n, seconds, marker="o", label=label)
        if len(group) >= 2:
            slope, intercept = np.polyfit(np.log(n), np.log(seconds), 1)
            fit = np.exp(intercept) * n ** slope
            ax.plot(n, fit, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(n[-1], fit[-1], f" n^{slope:.2f}", fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_\omega$")
    ax.set_ylabel("backend wall time / s")
    ax.set_title("Transport derivative backend scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_errors(rows: list[dict[str, str]], output_file: Path) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(label_for(row), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)
    for label, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: int(item["n_omega"]))
        n = np.asarray([int(item["n_omega"]) for item in group], dtype=float)
        signal_error = np.asarray([
            numeric(item, "order3_error", "order2_error")
            if int(item.get("max_order", "2") or 2) >= 3
            else numeric(item, "order2_error")
            for item in group
        ], dtype=float)
        matrix_error = np.asarray([
            numeric(item, "order3_T_rel_error", "order2_T_rel_error")
            if int(item.get("max_order", "2") or 2) >= 3
            else numeric(item, "order2_T_rel_error")
            for item in group
        ], dtype=float)
        axes[0].plot(n, signal_error, marker="o", label=label)
        axes[1].plot(n, matrix_error, marker="s", label=label)

    axes[0].set_title("Best available signal error")
    axes[0].set_ylabel("relative error")
    axes[1].set_title("Best available transport-matrix error")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_\omega$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dimer backend benchmark scaling.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dimer_backend_benchmarks/scaling_plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv_files)
    write_combined_csv(rows, args.output_dir / "combined_backend_scaling.csv")
    plot_timing(rows, args.output_dir / "backend_time_scaling.png")
    plot_errors(rows, args.output_dir / "backend_error_scaling.png")
    print(f"wrote {args.output_dir / 'backend_time_scaling.png'}")
    print(f"wrote {args.output_dir / 'backend_error_scaling.png'}")


if __name__ == "__main__":
    main()
