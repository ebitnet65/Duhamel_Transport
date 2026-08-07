# Iterative SAKE inverse-recovery benchmark

This directory preserves the August 2026 three-target synthetic inverse test
used to evaluate trust-region recentering of local SAKE charts.

## Model and controls

- mean site energy: `omega0 = 2.0 eV`;
- site asymmetry: `delta_omega = 0.18 eV`;
- dimensionless controls: `lambda_J = J/omega0` and
  `lambda_k = kappa/omega0`;
- initial reference: `J = kappa = 0`;
- physical bounds: `-0.15 <= J <= 0.15 eV` and
  `0 <= kappa <= 0.03 eV`;
- RP and NRP pathway states on a `20 x 20` recovery grid;
- third-order Duhamel transport with four-point quadrature.

The rephasing spectroscopy demonstration uses a separate `64 x 64` display
grid, so the rendered line shapes are not limited by the coarser grid used in
the inverse-recovery benchmark.

The three targets are `(-100, 12)`, `(80, 8)`, and `(-60, 20)` meV. Each
accepted trial point is checked by direct resolvent inversion. Internal
objective evaluations use the local third-order SAKE surrogate.

## Preserved artifacts

- `figures/`: normalized recovery, spectral-evolution, pathway-mixing, and
  animated-trajectory products;
- `multi_target_summary.csv`: numerical recovery and transport diagnostics;
- `multi_target_summary.tex`: compact manuscript table;
- `runs/*/iterations.csv`: complete accepted trust-region histories;
- `runs/*/summary.json`: run configuration and final diagnostics;
- `runs/*/accumulated_transport.npz`: global initial-to-recovered transport,
  direct target transport, and the product of local projected maps.

The stable pathway-mixing object is the global initial-to-recovered projection
stored as `accumulated_transport`. The separately stored
`composed_local_transport` is a closure diagnostic. Finite projected pathway
bases are not exactly transitive, so the product of local maps can differ
substantially from the global projection when out-of-basis residuals accumulate.

## Reproduction

From the repository root, run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false bash run_sake_iterative_suite.sh
```

For a narrative, cell-by-cell reproduction with the recovered-parameter table,
normalized publication figure, and final pathway-mixing heat maps, execute
`notebooks/04_iterative_sake_target_recovery.ipynb`.

The rephasing figures and movie can be regenerated with
`plot_sake_rephasing_evolution.py`; its default spectral grid is `64 x 64` and
values of at least `40 x 40` are recommended for presentation.

Raw generated outputs are written under the ignored
`dimer_backend_benchmarks/` directory. The selected results in this directory
are a version-controlled snapshot for provenance and manuscript preparation.
