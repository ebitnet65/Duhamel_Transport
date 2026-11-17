# Two-control spectroscopy calculation

This exploratory extension of the finite-difference JCP3 model uses the
control pair
\[
(\theta,\gamma_\phi),
\]
where \(\theta\) rotates the dephasing operator relative to the Hamiltonian
basis and \(\gamma_\phi\) sets its rate. The amplitude-relaxation rate is
held at \(\gamma_1=0.02\), with \(\omega_0=1\).

## Spectral calculation

The driver computes the complete rephasing response
\(S_R^{(3)}(\omega_3,t_2,\omega_1)\) on a square frequency grid. It sums the
two RWA rephasing pathways using the existing response operators and
resolvents. The preparation is the aligned-model stationary state at the same
\(\gamma_\phi\); at \(\theta=0\), this is \(|g\rangle\langle g|\) over the
tested rate range. The scans use
\[
\theta\in\{0,\pi/4,\pi/2\},\quad
\gamma_\phi\in\{0.05,0.10,0.20\},\quad
t_2\in\{0.2,1,5\},
\]
with \(\omega_1,\omega_3\in[0.25,1.75]\) and \(\eta=10^{-3}\).

Run from the repository root:

```bash
python calculate_two_control_spectroscopy.py --n-omega 61 --out two_control_spectroscopy
```

The output directory contains the full complex maps in `response_maps.npz`,
summary diagnostics in `summary.json`, and exploratory maps as PNG files.
The scalar deformation metric is the frequency-integrated complex L2
difference from the aligned-angle map at the same rate and waiting time,
normalized by the aligned-map L2 norm.

## State-generator geometry

For the geometric diagnostic, the model point is embedded as the real and
imaginary components of the stationary density matrix together with the real
and imaginary entries of the full Liouvillian matrix. The generator-coordinate
scale is one in the reduced units \(\omega_0=1\). The induced metric is the
Euclidean pullback metric of this declared embedding. Central finite
differences give the embedding Jacobian and Hessian; their tangential and
normal components give the Levi-Civita connection and Gaussian curvature.
This calculation does not evaluate the state-generator response two-form or a
spectral holonomy.

At \((\theta,\gamma_\phi)=(\pi/4,0.10)\), using the dimensionless
coordinates \((\theta,\gamma_\phi/0.10)\), the metric eigenvalues are
approximately \(0.08012\) and \(0.13073\), and the Gaussian curvature is
\(-0.62916\). The curvature changes by less than \(7\times10^{-6}\) as the
central-difference step is reduced from \(10^{-3}\) to \(3\times10^{-5}\).
Across the sampled interior grid
\(0.05\leq\theta\leq\pi/2-0.05\),
\(0.05\leq\gamma_\phi\leq0.20\), the metric remains positive definite;
the minimum sampled eigenvalue is about \(0.0202\). The maximum metric
condition number is about \(45.4\). The stationary states remain positive
and normalized, and the sampled Liouvillian spectral gap stays above
\(0.0202\).

## Preliminary result

At \\(\\theta=\\pi/2\\), changing \\(\\gamma_\\phi\\) from 0.10 to 0.05 or 0.20 changes the \\(t_2=1\\) map by about 10.4% or 12.6% in integrated relative L2 norm.\n\nAt \(\gamma_\phi=0.10\), the integrated relative map change between
\(\theta=\pi/2\) and the aligned map is about 0.374, 0.291, and 0.574 for
\(t_2=0.2,1,5\), respectively. The full maps show a distinct
waiting-time-dependent deformation. This supports retaining a short \(t_2\)
scan in the revision; it does not establish geometric transport of a spectrum
around a parameter-space loop.

At finite angle, the operator \(A_\theta\) has a transverse component in the
Hamiltonian basis. Thus varying \(\gamma_\phi\) changes both coherence decay
and the transverse dissipative action. The manuscript should describe this
control as the rate of the rotated dissipator and avoid interpreting it as a
pure linewidth parameter.
