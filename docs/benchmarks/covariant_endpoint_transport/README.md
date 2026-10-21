# Fixed-endpoint covariant-transport milestone

This directory archives the first end-to-end SAKE-DT benchmark in which a
path-independent two-dimensional spectrum determines a fixed endpoint and a
physical control-manifold geodesic is solved only afterward. It is a branch
milestone, not a final publication dataset.

## Scientific scope

The dimer controls are

\[
q=(J_{12},k_{12},\gamma_\phi),
\]

where detailed-balance transfer uses the pair of Lindblad operators
\(\sqrt{k_{1\to2}}|01\rangle\langle10|\) and
\(\sqrt{k_{2\to1}}|10\rangle\langle01|\), and collective dephasing uses
\(\sqrt{\gamma_\phi}(\sigma_z^{(1)}+\sigma_z^{(2)})\). The synthetic truth is

\[
q_{\mathrm{true}}=(0.025,0.014,0.022).
\]

The spectral inversion has no path or geodesic penalty. Once the endpoint is
recovered, it is frozen and the physical route from
\(p=(0.070,0.003,0.004)\) is obtained independently by discrete path-energy
minimization.

## Archived result

The deterministic three-by-three spectral benchmark produced:

- recovered endpoint
  \((0.02499999991,0.01399999936,0.02200000010)\);
- relative spectral residual \(1.26\times10^{-9}\);
- endpoint absolute error \(6.57\times10^{-10}\);
- full spectral-Jacobian rank, with condition number 23.79 at \(t_2=0.8\);
- maximum endpoint drift \(6.57\times10^{-10}\) across \(t_2=0.2,0.8\);
- full-rank joint waiting-time information with generalized eigenvalues
  \((5.4925\times10^7,1.2110\times10^6,6.6985\times10^3)\);
- straight-path energy 0.03681362 and geodesic energy 0.03502273, a 4.86%
  reduction;
- zero numerical geodesic endpoint error;
- maximum transported-frame Gram error below \(10^{-10}\).

The stable recovered endpoint and changing generalized observability spectrum
show two distinct effects: the physical model endpoint is invariant in this
correctly specified noiseless test, while the combinations of controls visible
to the experiment change with waiting time.

## Difference from JCPIV_SAKE

JCPIV_SAKE established the non-covariant local spectral/autodiff expansion and
endpoint-recovery workflow. It did not use the Liouvillian-induced physical
metric to discover a route between models and did not parallel-transport a
frame on the physical control manifold.

This milestone adds:

1. a fixed, path-independent spectral endpoint inversion;
2. the distinction between spectral information \(I=J_{\mathcal O}^T J_{\mathcal O}\)
   and the physical metric \(G\);
3. generalized physical-observability modes \(Iv=\chi Gv\);
4. a boundary-valued geodesic from the reference model to the recovered
   endpoint; and
5. metric-compatible parallel transport of control-manifold tangent frames
   along both the geodesic and straight comparison route.

This is therefore a geometric model-discovery benchmark, rather than merely a
change to the fitting method used in JCPIV_SAKE.

## Important limitation and next milestone

The transported objects here are tangent frames on the three-dimensional
control manifold. They are not yet the Liouville response-pathway frames used
to assemble the two-dimensional spectrum. For the square-root collapse-operator
embedding tested here, the intrinsic control manifold is flat; rate-coordinate
geodesics curve, but endpoint tangent-frame transport is effectively path
independent.

The next milestone is parallel transport of the actual local Liouville pathway
frame along the recovered-endpoint geodesic. It must:

- define the local pathway-frame connection or projection consistently with
  the SAKE/Duhamel representation;
- compose local pathway transport matrices in path order;
- preserve appropriate overlap, normalization, and subspace diagnostics;
- compare geodesic and straight-line pathway transport at the same fixed
  endpoint; and
- determine whether pathway holonomy or transported spectral subspaces retain
  path dependence even when the present tangent-frame embedding is flat.

## Reproduction

Run from the repository root with JAX 64-bit mode enabled:

```bash
python run_dimer_endpoint_recovery.py \
  --output docs/benchmarks/covariant_endpoint_transport/endpoint/recovery.json \
  --n-omega 3 --t2 0.8 --truth 0.025 0.014 0.022 \
  --seed 0 --random-starts 0

python run_dimer_waiting_time_robustness.py \
  --output-dir docs/benchmarks/covariant_endpoint_transport/waiting_time \
  --n-omega 3 --t2 0.2 0.8 --truth 0.025 0.014 0.022 \
  --seed 0 --random-starts 0

python benchmark_recovered_endpoint_transport.py \
  --endpoint-json \
    docs/benchmarks/covariant_endpoint_transport/endpoint/recovery.json \
  --reference 0.070 0.003 0.004 --vertices 9 \
  --output-dir docs/benchmarks/covariant_endpoint_transport/transport
```

The archive contains compact JSON summaries and compressed NumPy arrays:

- `endpoint/recovery.json`;
- `waiting_time/waiting_time_robustness.json` and `.npz`;
- `transport/transport_diagnostics.json` and `transport_paths.npz`.

The calculation was run and verified on the Mac Studio. The portable launchers
are prepared for BadShasta, but this archived record has not yet been reproduced
there.
