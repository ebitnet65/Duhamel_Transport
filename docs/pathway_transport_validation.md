# Liouville pathway-frame transport validation

This validation layer is intentionally independent of the transport
implementation. A transport adapter accepts a control-space path and an
initial ambient-space frame, then returns the endpoint frame. This separation
prevents a test from reproducing the same connection or integration error as
the implementation it is intended to check.

## Required checks

1. **Identity:** transport along a constant path leaves every frame unchanged.
2. **Composition:** forward transport followed by reverse-path transport
   recovers the initial frame.
3. **Overlap preservation:** the endpoint Gram matrix equals the initial Gram
   matrix in the appropriate endpoint metrics.
4. **Refinement:** endpoint subspaces converge when the same geometric
   polyline is subdivided.
5. **Rank:** singular values reveal collapse or loss of pathway directions.
6. **Internal gauge:** replacing a frame `B` by `B Q`, for nonsingular `Q`,
   cannot change the physical subspace. Principal angles, rather than raw frame
   differences, test this invariant.
7. **Coordinate representation:** under an ambient representation change
   `B'(q) = S(q) B(q)`, the endpoint transport must transform as
   `T' = S(q1) T S(q0)^-1`.
8. **Path dependence:** two paths with common endpoints are physically distinct
   only if a gauge-invariant diagnostic such as their endpoint principal angles
   or closed-loop holonomy remains nonzero after numerical refinement.

## Controlled validation

`test_pathway_transport_validation.py` uses exact segment exponentials for
constant skew connections. Commuting one-dimensional rotations give exact
identity, reversibility, metric preservation, gauge covariance, and refinement
invariance. Two noncommuting rotation generators produce different rank-one
endpoint subspaces for paths that traverse the same rectangular controls in
opposite orders. The nonzero principal angle survives arbitrary rescaling and
sign changes of the frame column, so it is not a basis-gauge artifact.

The controlled tests prove the correctness of the diagnostic definitions and
provide reference cases for a pathway transport adapter. They do **not** yet
prove that the production Liouville pathway connection is physically correct.
That requires binding the adapter to the production API and checking these
invariants on optical-Bloch and dimer pathway frames. In particular, a small
raw frame difference is not evidence of physical agreement, and a large raw
difference is not evidence of physical path dependence.

## Production acceptance targets

Thresholds must be scaled to solver tolerance and conditioning, but a first
double-precision target is:

- identity and gauge covariance below `1e-11`;
- forward/backward and Gram errors below `1e-8`;
- stable numerical rank across refinement;
- monotone principal-angle convergence with at least two refinements in the
  asymptotic regime;
- any claimed path dependence remains separated from the refinement error by
  at least one order of magnitude.

These are validation targets, not changes to the physical transport law.
