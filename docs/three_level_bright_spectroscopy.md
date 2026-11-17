# Three-level bright-state spectroscopy calculation

This calculation extends the finite-difference branch's full 2D response
workflow to a three-state V system with basis \(|G\rangle,|B_1\rangle,|B_2\rangle\).
The two bright states are the lower and upper eigenstates of the one-exciton
Hamiltonian
\[
H_1=\begin{pmatrix}\omega_1&J_{12}\\J_{12}&\omega_2\end{pmatrix},
\]
using the dimer site energies from the SAKE parameterization \(\omega_1=1\), \(\omega_2=1.18\),
and the \(J_{12}=0.045\) coupling used in the repository's dimer spectral diagnostic. Equal parallel site dipoles define the two transition
moments in the bright basis. The resulting energies are approximately
0.98938 and 1.19062.

Both bright states relax to \(|G\rangle\) and have state-specific pure
dephasing. The new internal-conversion channel is one-way,
\[
C_{21}=\sqrt{\kappa}\,|B_1\rangle\langle B_2|,
\]
so population transfers from the upper bright state to the lower state. The
default rates are \(\gamma_{B_1}=0.03\), \(\gamma_{B_2}=0.004\),
\(\gamma_{\phi,B_1}=0.02\), and \(\gamma_{\phi,B_2}=0.012\). The scan
uses \(\kappa=0,0.02,0.06\) and \(t_2=0.2,1,5,10\).

The Hilbert space deliberately omits the double-exciton state. This is a
three-level bright-state model with two ground-state transitions and
interbright population relaxation; it does not include excited-state
absorption. The driver computes complete complex RP and NRP maps and the
absorptive combination on a default \(81\times81\) grid from 0.78 to 1.38.
As in the two-level calculation, positive plotted frequencies correspond to
signed resolvent arguments \((-\omega_1,-\omega_3)\), and the displayed
absorptive signal is \(-\operatorname{Re}(S_R+S_{NR})\).

Run from the repository root:

```bash
python calculate_three_level_bright_spectra.py --n-omega 81 \\
  --out three_level_bright_spectra
```

The output includes channel-resolved and absorptive maps in
`three_level_response_maps.npz`, a model and numerical summary in
`summary.json`, and two exploratory PNG figures. Vectorized maps are
cross-checked against independent pointwise RP and NRP evaluations; the
maximum absolute discrepancy in the default calculation is about
\(1.2\times10^{-12}\).

This is a focused bright-state extension for the JCP3 revision. It uses the
dimer's one-exciton Hamiltonian and dipoles as a parameter source, but it does
not reproduce the four-level SAKE benchmark or its autodifferentiable
transport calculation.

## Finite-difference reconstruction

The exact RP and NRP maps are central-differenced with respect to \(\kappa\)
about \(\kappa_0=0.03\). The default selected step is \(h=0.005\); steps
0.0025 and 0.01 provide a derivative step-sensitivity check. First and second
map derivatives are combined into the local quadratic reconstruction
\[
S^{(2)}(\kappa)=S(\kappa_0)+(\kappa-\kappa_0)S'(\kappa_0)
+\tfrac12(\kappa-\kappa_0)^2S''(\kappa_0).
\]
At \(t_2=10\), its relative integrated L2 error is 0.328 at \(\kappa=0\) and
0.400 at \(\kappa=0.06\). The first-derivative maps change by about 0.98%
between steps 0.0025 and 0.005 over the waiting-time scan. These figures
separate the numerical FD derivative/reconstruction from the exact spectra
being reconstructed.
