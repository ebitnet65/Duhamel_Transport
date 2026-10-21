# Optical Bloch Geometry Test Branch

This branch isolates the next geometry upgrade from the current SAKE autodiff
dimer implementation.  The purpose is to validate the covariant machinery on a
model where the stationary state is known exactly.

## Model

The test system is a driven two-level optical Bloch model with amplitude
damping:

\[
H =
\begin{pmatrix}
0 & \Omega/2 \\
\Omega/2 & \Delta
\end{pmatrix},
\qquad
C = \sqrt{\gamma}\, |g\rangle\langle e|.
\]

The control vector is

\[
\lambda = (\Omega,\Delta,\gamma).
\]

The stationary state is represented by Bloch coordinates

\[
r(\lambda)
=
\left(
2\operatorname{Re}\rho_{ge},
2\operatorname{Im}\rho_{ge},
\rho_{ee}-\rho_{gg}
\right),
\]

with closed-form solution

\[
D = 4\Delta^2 + 2\Omega^2 + \gamma^2,
\]

\[
r(\lambda)
=
\left(
-\frac{4\Delta\Omega}{D},
\frac{2\Omega\gamma}{D},
-\frac{4\Delta^2+\gamma^2}{D}
\right).
\]

For the first geometry test, the generator coordinates are taken as

\[
h(\lambda) = (\Omega,\Delta,\gamma).
\]

This is deliberately simple.  It gives a controlled place to validate the
metric, two-form, and Christoffel-symbol machinery before applying the same
interfaces to the SAKE dimer.

## Geometry

The physical pullback metric is

\[
G_{\mu\nu}
=
\partial_\mu r\cdot \partial_\nu r
+
\partial_\mu h\cdot \partial_\nu h.
\]

The pulled-back two-form is

\[
F_{\mu\nu}
=
\partial_\mu r\cdot \partial_\nu h
-
\partial_\nu r\cdot \partial_\mu h.
\]

The Christoffel symbols are assembled from the metric:

\[
\Gamma^\alpha_{\mu\nu}
=
\frac{1}{2}G^{\alpha\beta}
\left(
\partial_\mu G_{\beta\nu}
+
\partial_\nu G_{\beta\mu}
-
\partial_\beta G_{\mu\nu}
\right).
\]

## Implemented Backends

The file `optical_bloch_geometry.py` currently provides:

- Closed-form \(r(\lambda)\) and \(\rho_{\mathrm{ss}}(\lambda)\).
- Symbolic non-autodiff \(G\), \(F\), and \(\Gamma\) from the closed-form
  \(r,h\) maps when SymPy is available.
- Finite-difference derivatives of the closed-form \(r,h\) maps.
- JAX autodiff geometry from the closed-form \(r,h\) maps.
- JAX autodiff geometry from the actual Liouvillian stationary-state solve.

The key validation is that the Liouvillian route,

\[
\mathcal L(\lambda)\rho_{\mathrm{ss}}(\lambda)=0,
\qquad
\operatorname{Tr}\rho_{\mathrm{ss}}=1,
\]

reproduces the same \(r\), \(G\), \(F\), and \(\Gamma\) as the closed-form route.

## Commands

Run the focused tests:

```bash
pytest -q test_optical_bloch_geometry.py
```

Generate a diagnostic output directory:

```bash
python demo_optical_bloch_geometry.py \
  --omega 0.35 \
  --delta 0.12 \
  --gamma 0.08 \
  --output-dir optical_bloch_geometry_demo
```

The demo writes:

- `geometry_summary.json`
- `optical_bloch_geometry.npz`

## Parallel Transport Prototype

`Liouville_covariant_transport.py` implements model-independent
Levi-Civita parallel transport of tangent vectors and complete frames along a
supplied piecewise-linear control path. Each frame column satisfies

\[
\frac{dV^\alpha}{ds}
=-\Gamma^\alpha_{\mu\nu}(\lambda(s))
  \dot\lambda^\mu V^\nu.
\]

The optical Bloch tests check metric inner-product preservation, recovery on
the reversed path, and agreement between symbolic and Liouvillian-autodiff
Christoffel backends. Run them with:

```bash
pytest -q test_liouville_covariant_transport.py
```

This prototype transports control-manifold tangent frames. Mapping those
frames into model-specific Liouville pathway bases remains a separate layer.

### New-model robustness check

`dimer_covariant_geometry.py` applies the same transport layer to the existing
four-level exciton-dimer Liouvillian without modifying the dimer model. The
test manifold uses controls
\((\omega_2-\omega_1,J_{12},k_{12})\), steady-state density coordinates, and
the real and imaginary Liouvillian entries as generator coordinates. The
detailed-balance transfer rates give a nonconstant metric and nonzero
connection, avoiding the trivial flat geometry that would result from the
unpumped dimer stationary state with \(h=\lambda\).

```bash
pytest -q test_dimer_covariant_transport.py
```

## Discrete Geodesics and Path Comparison

`Liouville_geodesics.py` minimizes the midpoint-discretized energy at fixed
endpoints:

\[
E[\gamma]
=
\frac{1}{2}\int_0^1
\dot{\lambda}^{\mu}
G_{\mu\nu}(\lambda)
\dot{\lambda}^{\nu}
\,ds.
\]

The dimer comparison uses eight vertices and checks the optimized path against
coordinate-space straight interpolation. In the test regime the geodesic
reduces energy by about \(1.65\times10^{-3}\), its largest coordinate-space
departure from the straight path is about \(2.08\times10^{-3}\), and the two
paths produce endpoint frames separated by about \(1.78\times10^{-3}\) in
Frobenius norm. Both transports preserve metric inner products.

```bash
pytest -q test_liouville_geodesics.py test_dimer_geodesic_transport.py
```

A later model adapter can map the transported tangent frame into local
Liouville pathway frames.

### Three-control collapse-operator benchmark

The next dimer benchmark uses
\(\lambda=(J,k,\gamma_\phi)\) at fixed detuning and temperature. Transfer
collapse operators obey detailed balance,

\[
C_{1\to2}=\sqrt{k e^{-\beta\Delta/2}}\,|01\rangle\langle10|,
\qquad
C_{2\to1}=\sqrt{k e^{+\beta\Delta/2}}\,|10\rangle\langle01|,
\]

and collective dephasing uses
\(C_\phi=\sqrt{\gamma_\phi}(\sigma_z^{(1)}+\sigma_z^{(2)})\).
The physical generator embedding contains these square-root collapse operators
directly. The optimized nine-vertex path reaches the specified truth endpoint
and lowers energy by about 5.59 percent. Both rate coordinates depart visibly
from straight interpolation; \(J\) remains linear. This embedding is
intrinsically flat, so endpoint parallel transport is path independent even
though the geodesic is curved in the original rate coordinates.

```bash
pytest -q test_dimer_three_control_geodesic.py
```
