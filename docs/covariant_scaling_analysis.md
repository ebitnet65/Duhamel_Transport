# Covariant Geometry Scaling Analysis

This benchmark measures a fixed seven-vertex covariant pipeline as the number
of controls grows from 2 to 16. Every dimension uses the same smooth nonlinear
coordinate-embedding family, so these timings measure algorithmic scaling and
not changes in Hilbert- or Liouville-space model size.

Run:

```bash
python benchmark_covariant_scaling.py \
  --dimensions 2 3 4 6 8 10 12 16 \
  --vertices 7 \
  --repeats 100 \
  --output-dir covariant_scaling_benchmark
```

## Results

| controls | compile (s) | Christoffel (ms) | geodesic (s) | objective evaluations | transport (s) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.136 | 0.031 | 0.056 | 231 | 0.019 |
| 3 | 0.247 | 0.064 | 0.108 | 352 | 0.126 |
| 4 | 0.188 | 0.064 | 0.145 | 525 | 0.072 |
| 6 | 0.212 | 0.062 | 0.569 | 775 | 0.057 |
| 8 | 0.216 | 0.071 | 0.720 | 984 | 0.064 |
| 10 | 0.240 | 0.080 | 0.962 | 1224 | 0.107 |
| 12 | 0.246 | 0.080 | 1.477 | 1525 | 0.069 |
| 16 | 0.216 | 0.153 | 2.687 | 2106 | 0.081 |

Log-log fits over this range give approximate empirical exponents:

- warm metric evaluation: \(n^{0.28}\)
- warm Christoffel evaluation: \(n^{0.53}\)
- geodesic objective evaluations: \(n^{1.04}\)
- total geodesic solve time: \(n^{1.89}\)
- frame transport time: \(n^{0.36}\)

The small compiled tensors in this range make the local timing exponents
latency dominated; they should not be interpreted as asymptotic tensor
complexities. The geodesic result is more informative: with a fixed number of
vertices, finite-difference L-BFGS-B introduces a number of objective calls
approximately proportional to control dimension, while each path-energy call
also becomes more expensive. Endpoint/path optimization is therefore already
the dominant cost by 6--8 controls.

All dimensions converged, metric condition numbers remained below 5.4, and
the largest transported-frame Gram error was below \(7.4\times10^{-11}\).

## Implication

The present implementation is practical through moderate control dimension,
but the finite-difference optimizer is the first scaling bottleneck. The next
production improvement should differentiate the discrete path energy with JAX
and provide its gradient directly to the optimizer. That removes the linear
finite-difference multiplier in objective evaluations. Batched metric
evaluation along path vertices is the next likely improvement after that.

This benchmark does not measure scaling with Hilbert-space dimension. A
separate model-size sweep is required for the stationary-state solve and
Liouvillian coordinate construction.
