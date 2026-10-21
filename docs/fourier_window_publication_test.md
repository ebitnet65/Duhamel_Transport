# Fourier-window robustness: publication test record

## Result

The acquisition-window benchmark demonstrates that the inferred transfer rate
\(k_{12}\) and coherent coupling \(J_{12}\) are sensitive to Fourier-window
apodization even while the ideal-model Jacobian remains full rank. In the
archived sweep, the exponential apodization strength ranges from 0 to 0.01 at
fixed \(q_{\mathrm{true}}=(0.025,0.014,0.022)\), \(t_2=0.8\), and a
\(3\times3\) spectral grid.

At the largest tested strength, the recovered controls are approximately

\[
(J_{12},k_{12},\gamma_\phi)
=(0.02046,0.00010,0.02754).
\]

Relative to truth, \(J_{12}\) decreases by about 18%, \(k_{12}\) decreases by
more than 99% and reaches its inversion lower bound, and \(\gamma_\phi\)
increases by about 25%. The strong loss of inferred transfer is therefore not
an identifiability-rank failure. It is a compensating endpoint bias caused by
an observation feature outside the ideal fitted Liouvillian model.

The sampled acceptance window is \([0,0.001]\). At strength 0.002, the first
failed criterion is the relative tangent-orthogonal residual. This residual is
the appropriate warning signal: the optimizer continues to produce a plausible
endpoint after the ideal model has ceased to explain the distorted spectrum.

## Interpretation

The result supports three practical conclusions.

1. Fourier-window and acquisition parameters must either be included in the
   observation model or calibrated independently before interpreting recovered
   \(J_{12}\) and \(k_{12}\).
2. A successful optimizer and full-rank spectral Jacobian do not establish
   model adequacy. The residual component orthogonal to the model tangent space
   must be reported.
3. Geodesic and pathway-transport results inherit endpoint bias. Outside the
   accepted window, route trajectories remain numerically well defined but no
   longer represent a reliably identified physical endpoint.

## Mathematical definition of apodization

Apodization is multiplication of a time-domain response by a window before
Fourier transformation. For the two-dimensional response
\(R(t_3,t_1;t_2)\), define

\[
R_w(t_3,t_1;t_2)=w(t_3,t_1)R(t_3,t_1;t_2),
\]

and

\[
S_w(\omega_3,\omega_1;t_2)
=\mathcal F_{t_3,t_1}\!\left[R_w(t_3,t_1;t_2)\right].
\]

The production benchmark uses the separable symmetric exponential window

\[
w_{\alpha_3,\alpha_1}(t_3,t_1)
=\exp[-\alpha_3|t_3|-\alpha_1|t_1|].
\]

For the plotted sweep, \(\alpha_3=\alpha_1=\alpha\), so

\[
w_\alpha(t_3,t_1)=\exp[-\alpha(|t_3|+|t_1|)].
\]

The parameters \(\alpha_1\) and \(\alpha_3\) have inverse-time units. At
\(\alpha=0\), the time response and spectrum are unchanged. Increasing
\(\alpha\) preferentially suppresses long-time contributions and therefore
broadens frequency-domain structure.

Under the continuous Fourier convention

\[
\widetilde f(\omega)=\int_{-\infty}^{\infty}f(t)e^{i\omega t}\,dt,
\]

the transform of a one-dimensional exponential window is

\[
\mathcal F[e^{-\alpha|t|}](\omega)
=\frac{2\alpha}{\alpha^2+\omega^2}.
\]

Consequently, multiplication by the exponential window in time corresponds to
convolution with a Lorentzian kernel along each frequency axis:

\[
S_\alpha=S_0*\widetilde w_\alpha.
\]

This explains why apodization can broaden and merge spectral features that the
ideal inversion may otherwise attribute to changes in \(J_{12}\), \(k_{12}\),
or \(\gamma_\phi\).

### Discrete implementation

The code stores arrays in \((\omega_3,\omega_1)\) and \((t_3,t_1)\) order. It
uses the centered unitary NumPy transform pair

\[
R_{mn}=\operatorname{fftshift}\!\left[
\operatorname{IFFT2}\!\left(\operatorname{ifftshift}S\right)
\right]_{mn},
\]

\[
S_{jk}=\operatorname{fftshift}\!\left[
\operatorname{FFT2}\!\left(\operatorname{ifftshift}R\right)
\right]_{jk},
\]

with `norm="ortho"`. The exponential operation is therefore

\[
R^{(\alpha)}_{mn}
=\exp[-\alpha_3|t_{3,m}|-\alpha_1|t_{1,n}|]R_{mn},
\]

followed by the centered forward FFT2. This discrete definition, including the
absolute values, axis order, normalization, and grid construction, is the
definition used to generate the archived production figures.

## Exact production configuration

- artifact family: symmetric separable exponential apodization;
- strengths: `0, 0.0005, 0.001, 0.002, 0.005, 0.01`;
- truth controls: `J_12=0.025`, `k_12=0.014`, `gamma_phi=0.022`;
- reference controls: `J_12=0.070`, `k_12=0.003`, `gamma_phi=0.004`;
- waiting time: `t2=0.8`;
- frequency grid: three equally spaced points from 0.86 to 1.32 on each axis;
- geodesic path: four vertices;
- pathway refinement factors: 1 and 2;
- deterministic optimizer starts and no detector noise.

The input JSON/NPZ archive is `fourier_artifact_robustness_benchmark/`.

## Reproduction

Generate both final figures and their audit data:

```bash
python generate_fourier_window_publication_artifacts.py \
  --input-dir fourier_artifact_robustness_benchmark \
  --output-dir publication_artifacts/fourier_window
```

Run validation:

```bash
pytest -q test_fourier_window_publication.py
```

The production driver writes PDF and 600-dpi PNG figures, a CSV containing the
plotted control estimates, and an NPZ containing every plotted route.

## Scope and limitations

These figures document a controlled numerical sensitivity test, not yet a
complete experimental error model. The spectral grid is deliberately small,
the time signal is obtained from the periodic band-limited inverse transform of
that grid, and the plotted production sweep varies only symmetric exponential
apodization. Causal pulse generation, finite acquisition cutoffs, phase error,
sampling, detector noise, and realistic experimental grids require separate
production sweeps before quantitative experimental uncertainty claims can be
made.
