# TLS theta–gamma-phi response geometry surface

This finite-difference calculation returns to the rotated-dephasing TLS and
samples the two controls \((\theta,\gamma_\phi)\) on a 5-by-5 grid:
\[
\theta\in\{0,\pi/8,\pi/4,3\pi/8,\pi/2\},\qquad
\gamma_\phi\in\{0.05,0.075,0.10,0.15,0.20\}.
\]
It evaluates full complex RP and NRP maps and their absorptive combination at
\(t_2=0.2,1,5\). Central finite differences of the exact maps estimate
\(\partial_\theta S\) and \(\partial_{\gamma_\phi/0.1}S\) for every
surface node and pathway sector. At \((\pi/4,0.1)\), a two-coordinate
second-order Taylor map is compared with the exact maps over the surface.

The state-generator embedding is the concatenation of the real and imaginary
parts of the stationary density matrix and full Liouvillian, with generator
scale one in reduced units \(\omega_0=1\). The induced metric and its
Levi-Civita connection give Gaussian curvature \(K=-0.62916\) at the
reference point.

For a geometric response diagnostic, the metric-raised gradients of the RP,
NRP, and absorptive spectral maps at the reference point are parallel-
transported around this rectangle in coordinates
\((\theta,\gamma_\phi/0.1)\):
\[
(\pi/4,1)\to(\pi/4+0.70,1)\to
(\pi/4+0.70,1.90)\to(\pi/4,1.90)\to(\pi/4,1).
\]
The tangent-space holonomy is
\[
H=\begin{pmatrix}
0.998152&-0.008292\\
0.011319&1.001757
\end{pmatrix}.
\]
It preserves the reference metric to \(4.2\times10^{-9}\); reverse
transport composed with forward transport differs from identity by
\(2.7\times10^{-9}\). The transported spectral-sensitivity fields change
by about 0.95% in integrated metric norm for each waiting time and pathway
sector. The reference-node vectorized RP/NRP values agree with independent
pointwise calculations to 0 and 1.8e-15, respectively. Integration convergence from 32 to 64 steps per edge is
\(1.5\times10^{-8}\).

This is holonomy of the response-sensitivity vectors under the Levi-Civita
connection of the state-generator embedding. It is not a path-dependent
static spectrum, and it is not transport under the separate
propagator-derived form \((\partial_\mu\mathcal G)\mathcal G^{-1}\).
A static spectrum at a given \((\theta,\gamma_\phi)\) remains its
endpoint response.

Run from the repository root:

```bash
python calculate_tls_theta_gamma_surface.py --n-omega 51 --out tls_theta_gamma_surface
```

The output includes full channel maps, finite-difference derivatives and
reconstructions in `tls_theta_gamma_surface.npz`, a summary of geometry and
error checks in `summary.json`, and figures for the spectra, curvature, and
transported sensitivity. The largest relative integrated L2 error of the
second-order reconstruction on this sampled surface is about 4.18, so the
Taylor map is local; exact endpoint maps are retained.
