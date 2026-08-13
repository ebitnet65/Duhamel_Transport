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

## Next Upgrade

The next code step should be a geodesic solver using \(G_{\mu\nu}(\lambda)\).
For an endpoint \(q\), the path \(\gamma_{pq}\) should minimize

\[
E[\gamma]
=
\frac{1}{2}\int_0^1
\dot{\lambda}^{\mu}
G_{\mu\nu}(\lambda)
\dot{\lambda}^{\nu}
\,ds.
\]

That solver can then feed local pathway-frame transport along the physical
geodesic rather than along a straight line in parameter space.
