# Iterative SAKE wall-clock benchmark

This benchmark separates the cost of one directly inverted RP+NRP response
from the additional wall time of each accepted trust-region step.  It uses the
same asymmetric dimer, three targets, `20 x 20` recovery grid, third-order
Duhamel expansion, and four-point quadrature as the inverse-recovery benchmark.
Measurements were made on a Mac Studio with an Apple M2 Max processor (12 CPU
cores: 8 performance and 4 efficiency cores) and 64 GB of unified memory.  The
calculations used the JAX CPU backend with Python 3.13.5, JAX 0.11.0, NumPy
2.1.3, and SciPy 1.15.3.  Absolute times are hardware-dependent; the component
fractions identify the relevant bottleneck.

## Measurements

The instrumented driver records:

- one direct target-response evaluation;
- one-time construction of the initial local chart;
- surrogate optimization within each accepted step;
- direct inversion used to validate the proposed step;
- projection of the exact local transport map; and
- construction of the third-order chart centered at the accepted point.

Across the three targets, one direct response required `2.13--2.20 s`, the
initial chart required `8.40--9.08 s`, and an accepted iterative step required
`11.16--12.13 s` on average.  The complete six-to-eight-step recoveries required
`72.81--89.29 s`, excluding derivative-cache loading and the initial chart.

The mean iterative-time fractions were:

- local-chart rebuild: `79.6%`;
- direct trial validation: `19.9%`;
- surrogate optimization: `0.39%`; and
- transport projection: `0.014%`.

The timing therefore identifies chart construction, rather than optimization
or transport-matrix projection, as the principal target for performance work.
The direct response and each direct validation have essentially the same cost,
as expected; the additional factor of roughly five per accepted step comes
from rebuilding a third-order chart about the new center.

## Artifacts

- `figures/iterative_wall_time_breakdown.pdf`: publication vector figure;
- `figures/iterative_wall_time_breakdown.png`: high-resolution preview;
- `iterative_wall_time_summary.csv`: cross-target timing summary; and
- `benchmark_environment.json`: hardware and software environment; and
- `runs/*`: complete per-step timings and JSON summaries.

Regenerate the plot with `plot_sake_iterative_timing.py` after running the
instrumented `fit_sake_dimer_iterative.py` for the three targets.
