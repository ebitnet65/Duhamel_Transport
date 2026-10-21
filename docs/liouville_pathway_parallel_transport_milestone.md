# Liouville Pathway-Frame Parallel-Transport Milestone Report

## Executive summary

This milestone extends the fixed-endpoint covariant SAKE-DT workflow from
parallel transport of control-manifold tangent frames to discrete parallel
transport of ordered Liouville response-pathway frames.

For the three-control exciton dimer,

\[
q=(J_{12},k_{12},\gamma_\phi),
\]

the endpoint is first recovered from a path-independent two-dimensional
spectrum. The recovered endpoint is then held fixed while the same ordered
Liouville pathway frame is transported along two routes: the physical-metric
geodesic and coordinate-space straight interpolation.

The calculation resolves a small but converged route dependence in the
transported pathway frame. The endpoint-aligned frame difference and route
holonomy distance are both \(2.747\times10^{-3}\), while the independently
evaluated endpoint spectra agree exactly. This is the desired conceptual
separation: the truth observable fixes the endpoint, whereas the route controls
how local Liouville pathway structure is compared between models.

The full repository validation after integration is:

```text
113 passed, 1 skipped
```

All calculations reported here were run on the Mac Studio. BadShasta has not
yet been used for this milestone.

## Relation to JCPIV_SAKE

JCPIV_SAKE established local spectral/autodiff expansion, Duhamel response
construction, and endpoint recovery without parallel transport on a physical
control manifold. The subsequent fixed-endpoint covariant milestone added the
Liouvillian-induced physical metric, geodesic discovery, and tangent-frame
parallel transport, but did not yet transport the Liouville response pathways.

The present milestone adds a new layer:

1. construct an ordered local frame from Liouville pathway-state columns;
2. orthonormalize the frame using SVD/Löwdin factors;
3. align adjacent frames using the unitary polar factor of their overlap;
4. compose the local maps in path order;
5. compare pathway transport along the geodesic and straight route at the same
   fixed endpoint; and
6. test gauge covariance, rank stability, Gram preservation, reversibility,
   refinement convergence, and closed-loop holonomy.

This is more than a modified fitting procedure. Endpoint inversion and pathway
transport answer different questions: the inversion identifies which model
matches the observation, while pathway transport describes how its local
response organization is related to that of the reference model.

## Mathematical and numerical construction

Let \(B_i\) denote the matrix whose columns are the selected Liouville pathway
states at the \(i\)-th point on a control-space path. Each full-rank frame is
orthonormalized to \(Q_i\). The adjacent-frame overlap is

\[
M_i=Q_{i+1}^{\dagger}Q_i.
\]

The canonical discrete transport step uses the unitary polar factor of
\(M_i\). The cumulative coefficient map is the path-ordered product of these
local steps. Raw projection maps are retained separately to diagnose finite-step
contraction and subspace loss.

This construction provides a stable discrete connection on a sequence of
overlapping subspaces. It explicitly separates:

- the ambient transported frame, which is the gauge-covariant physical object;
- coefficient matrices, which transform with the chosen local pathway gauge;
- principal angles and overlap singular values, which measure local subspace
  motion; and
- Gram and unitarity residuals, which diagnose numerical preservation.

## Fixed-endpoint dimer benchmark

The benchmark uses the archived recovered endpoint and reference model from the
preceding covariant milestone. Four ordered pathways are transported: two RP
and two NRP pathways. The current frame is constructed from diagonal-frequency,
after-\(U_2\) pathway states on a three-point frequency grid. Refinement factors
1, 2, and 4 are compared.

| Quantity | Straight route | Geodesic route |
|---|---:|---:|
| Physical path-energy reduction | — | 4.618% |
| Coarse-to-fine map error | \(2.388\times10^{-4}\) | \(2.355\times10^{-4}\) |
| Next refinement map error | \(5.969\times10^{-5}\) | \(5.889\times10^{-5}\) |
| Off-diagonal pathway mixing fraction | \(2.218\times10^{-6}\) | \(2.260\times10^{-6}\) |
| Maximum local principal angle | 0.01919 rad | 0.01951 rad |
| Maximum cumulative Gram error | \(\sim10^{-14}\) | \(\sim10^{-14}\) |

Cross-route results at the common endpoint are:

```text
endpoint aligned-frame difference = 2.747e-3
route holonomy distance           = 2.747e-3
endpoint spectral difference      = 0
```

Halving the path step reduces the map error by approximately a factor of four
for both routes. This is consistent with second-order convergence and supports
the interpretation that the observed route difference is resolved rather than
a step-size artifact.

## Independent validation

### Analytically controlled connection tests

The validation layer establishes the following properties on controlled skew
connections:

- constant-path identity;
- exact forward/reverse composition;
- metric-overlap preservation;
- internal gauge invariance;
- coordinate-representation covariance;
- explicit rank-loss detection; and
- genuine path dependence for noncommuting connection components.

### Production pathway-transport tests

The production implementation was checked empirically for:

```text
forward/backward error          = 5.48e-16
maximum local unitarity error   = 7.14e-16
maximum cumulative Gram error   = 7.76e-16
refinement map error            = 2.74e-3 -> 6.88e-4
closed-loop eigenvalues         = 0.999406 +/- 0.034461 i
```

The closed-loop spectrum, trace, and determinant are invariant under arbitrary
local unitary changes of pathway basis, as required for a gauge-independent
holonomy diagnostic.

## Scientific interpretation

The benchmark supports three conclusions.

First, the endpoint and route remain logically separate. Both routes reach the
same recovered model and therefore produce the same endpoint spectrum.

Second, the pathway frame retains information about the route. Even for a
control embedding whose tangent-frame transport is effectively path independent,
the ordered Liouville pathway subspace develops a small nonzero relative
holonomy between the geodesic and straight routes.

Third, waiting-time and pathway geometry provide information beyond endpoint
fit quality. The endpoint spectrum alone cannot report how pathway character,
mixing, or local response subspaces evolved between the reference and target
models.

## Limitations

The current result should not be overstated.

1. The transport is a canonical discrete subspace construction based on polar
   alignment. It has not yet been derived from the Levi-Civita connection of
   the physical control metric or from a unique continuous Liouville-bundle
   connection.
2. The selected pathway frame uses diagonal-frequency, after-\(U_2\) pathway
   states. The endpoint spectrum is evaluated independently on the full
   \(3\times3\), after-\(G_3\) grid. A full two-dimensional pathway-state frame
   is the stronger production benchmark.
3. Constant pathway rank and adequate overlap between adjacent subspaces are
   required. Rank loss and nearly orthogonal neighboring subspaces correctly
   stop the calculation rather than define a unique transport.
4. Quantitative results depend on pathway stage, frequency sampling, ambient
   inner product, and pathway ordering. These are presently model choices, not
   derived invariants.
5. Individual coefficient maps are gauge dependent. Only covariant ambient
   frames and gauge-invariant comparisons such as principal angles and holonomy
   spectra should be interpreted physically.
6. Refinement is diagnostic rather than adaptive.
7. The production path validator currently rejects an all-constant sampled
   path. Constant-path identity is proven in the controlled validation layer,
   but should also be supported or explicitly specified by the production API.
8. The benchmark is noiseless and uses an ideal impulsive spectrum.

## Recommended next milestones

1. Extend the transported frame to the full two-dimensional after-\(G_3\)
   pathway-state grid and test convergence with respect to both frequency axes.
2. Define a physically motivated weighted inner product or metric on pathway
   state space and determine how the connection changes with that choice.
3. Relate the polar-overlap discrete connection to an explicit continuous
   connection and to derivatives supplied by SAKE/JAX.
4. Resolve the constant-path production behavior and add adaptive refinement
   based on principal angles and overlap conditioning.
5. Repeat the route comparison across waiting times and endpoint uncertainty
   samples to determine whether pathway holonomy is statistically robust.
6. Reproduce the archived benchmark on BadShasta and compare numerical output,
   JAX backend, runtime, and conditioning with the Mac Studio record.

## Reproduction

Run the focused milestone through the portable launcher:

```bash
scripts/run_dimer_pathway_transport_milestone.sh \
  OUTPUT_DIR \
  docs/benchmarks/covariant_endpoint_transport/endpoint/recovery.json \
  --vertices 7 \
  --n-omega 3 \
  --refinement-factors 1 2 4
```

The launcher writes:

- `runtime_metadata.json`;
- `run.log`;
- `pathway_transport_diagnostics.json`; and
- `pathway_transport_arrays.npz`.

Repository-wide verification:

```bash
pytest -q
```

Expected result for this milestone:

```text
113 passed, 1 skipped
```
