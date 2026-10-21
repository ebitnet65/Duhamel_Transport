# Model-discrepancy robustness and uncertainty window

## Definition

The synthetic endpoint remains fixed while a deterministic acquisition layer
applies symmetric exponential apodization in the conjugate time domain. The
artifact strength `a` means

```text
exp[-a (|t3| + |t1|)].
```

Each distorted spectrum is then fitted with the ideal, unapodized dimer
observable. At the fitted endpoint, the residual is decomposed with the SVD of
the ideal observable Jacobian:

```text
r = r_parallel + r_orthogonal,
r_parallel in range(J),
r_orthogonal perpendicular to range(J).
```

The parallel component is locally explainable by parameter displacement. The
orthogonal component is locally out of model. At a converged least-squares
solution most remaining residual is expected to be orthogonal, so reliability
uses its magnitude relative to the observed spectrum, not its fraction of the
post-fit residual alone.

## Explicit empirical thresholds

The initial driver defaults are:

- relative total residual <= `0.05`;
- relative orthogonal residual <= `0.02`;
- orthogonal fraction <= `1.0` (reported but not restrictive by default);
- Euclidean endpoint bias <= `0.005`;
- local physical endpoint bias <= `0.25`;
- Jacobian condition number <= `1e6`;
- full control rank required.

These are declared empirical acceptance criteria, not coverage probabilities
or mathematical error bounds. The uncertainty window is the largest contiguous
prefix of the sampled artifact-strength grid, beginning at zero, for which all
replicates pass every criterion. A finer strength grid is required to localize
its boundary.

## Noise-free smoke result

Command:

```bash
python run_dimer_model_discrepancy.py \
  --output-dir /tmp/dimer_discrepancy_smoke \
  --n-omega 2 \
  --artifact-strength 0 0.002 0.01 \
  --random-starts 0
```

| Artifact strength | Endpoint bias | Physical bias | Relative residual | Relative orthogonal residual | Jacobian condition | Reliable |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 4.75e-11 | 3.04e-10 | 3.62e-11 | 5.72e-16 | 25.73 | yes |
| 0.002 | 1.29e-3 | 6.68e-3 | 9.39e-3 | 9.39e-3 | 25.57 | yes |
| 0.01 | 4.31e-3 | 1.86e-2 | 4.78e-2 | 4.78e-2 | 24.92 | no |

On this deliberately small grid, the resolved reliable window is `[0, 0.002]`.
The first failure at `0.01` is the relative orthogonal residual, rather than
loss of rank, poor conditioning, or excessive endpoint bias. This is a useful
model-discrimination outcome: the ideal model still finds a plausible endpoint
but cannot absorb the acquisition distortion into its three physical controls.

The reported physical uncertainty RMS is a local residual-scale proxy,

```text
sqrt(trace(G Cov_local)),
```

where `Cov_local` uses the orthogonal residual RMS and the pseudoinverse of the
information matrix. It is not a posterior credible interval. The API accepts
replicate seeds so explicit noise ensembles can replace this proxy later.

## Outputs

The driver writes:

- `model_discrepancy_robustness.json`: configuration, artifact metadata,
  complete inversion records, thresholds, failures, and diagnostics;
- `model_discrepancy_robustness.npz`: compact arrays for plotting and route or
  pathway-transport propagation.

Optional downstream diagnostics can be injected into the generic analyzer to
track geodesic, pathway-frame, or holonomy changes. Those changes are norms
relative to the fixed truth endpoint and do not alter the endpoint fit.
