#!/usr/bin/env python3
"""Regenerate every publication artifact for the Fourier-window benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_figure_fourier_window_paths import generate as generate_paths
from generate_figure_fourier_window_sensitivity import generate as generate_sensitivity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("fourier_artifact_robustness_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("publication_artifacts/fourier_window"))
    args = parser.parse_args(argv)
    generate_sensitivity(args.input_dir, args.output_dir)
    generate_paths(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
