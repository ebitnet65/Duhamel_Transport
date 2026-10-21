# Covariant SAKE-DT: Stand-Alone Paper Plan

## Publication separation

The manuscript currently under review as **JCPIV_SAKE** is the baseline and
must remain scientifically and computationally intact. The new paper should
not be framed as adding parallel transport to that method. Its subject is a
different problem:

> How can a spectroscopic inverse problem be formulated and solved on the
> physical manifold induced by a differentiable open-system Liouvillian, and
> how can local response/pathway information be compared covariantly between
> the discovered models?

JCPIV_SAKE supplies a local spectral/autodiff expansion and target-recovery
engine. Covariant SAKE-DT uses differentiable Liouvillian models to construct a
global control-manifold geometry, separates endpoint inference from route
discovery, and studies identifiability, observation dependence, and model
adequacy.

## Proposed identity

Working title:

**Covariant SAKE-DT: Geodesic Model Discovery and Parallel Transport on
Liouvillian Control Manifolds**

Alternative title emphasizing spectroscopy:

**Physical Model Manifolds for Nonlinear Spectral Inversion and Covariant
Pathway Transport**

One-sentence claim:

> A differentiable Liouvillian induces both symmetric and antisymmetric
> geometry on model-control space, allowing a fixed spectral endpoint to be
> inferred independently of the path and then connected to a reference model
> by a physical geodesic along which local response frames can be transported
> covariantly.

## Stronger scientific center: spectroscopic observability geometry

Endpoint fitting alone is not sufficient to distinguish this work from a
clever inverse procedure. The stronger physical question is:

> Which intrinsic directions of an open-quantum model manifold are visible to
> a nonlinear spectrum, how does that visibility change with waiting time, and
> which experimental protocol best resolves the physical model?

The intrinsic metric \(G^{\mathrm{phys}}\) and spectral information metric
\(I(t_2)\) define the generalized eigenproblem

\[
I_{\mu\nu}(q;t_2)v_a^\nu
=\chi_a(t_2)G^{\mathrm{phys}}_{\mu\nu}(q)v_a^\nu.
\]

The eigenvectors \(v_a\) are model-deformation modes orthonormal in the
Liouvillian metric. The generalized eigenvalues \(\chi_a\) measure how strongly
each physical mode is visible in the 2D spectrum. Unlike raw parameter
covariance, this spectrum is invariant under smooth control-coordinate changes.

This construction produces physical outputs beyond fitted parameters:

- waiting-time-dependent observable and dark model modes;
- coordinate-invariant conditioning and effective observable dimension;
- rotations or crossings of observable modes as population and coherence
  pathways evolve during \(t_2\);
- optimal waiting times or multi-time designs selected relative to
  \(G^{\mathrm{phys}}\);
- a test of whether an added model term opens a genuinely new observable
  direction or merely duplicates an existing one;
- endpoint uncertainty and geodesic bundles expressed in physical rather than
  arbitrary parameter units.

Equivalently, the coordinate-invariant operator

\[
\mathcal K(t_2)=G^{-1/2}I(t_2)G^{-1/2}
\]

has eigenvalues \(\chi_a\). Information from several waiting times adds,

\[
I_{\mathrm{joint}}=\sum_j I(t_{2,j}),
\]

so the theory predicts which waiting-time set closes spectral blind directions
most efficiently. This makes the method an experimental-design and physical-
observability framework; endpoint recovery becomes its validation and use case.

Where a concrete cyclic control observable exists, the antisymmetric two-form
\(F\) can provide an independent geometric prediction through curvature flux
or holonomic response. Such a prediction would test the inferred geometry
rather than merely its fitted spectrum. Transported basis rotations alone must
not be presented as measurable memory of a unique stationary state.

## Distinction from JCPIV_SAKE

| Question | JCPIV_SAKE baseline | Covariant SAKE-DT paper |
|---|---|---|
| Primary problem | Local response expansion and iterative target recovery | Global inverse discovery and transport on a model manifold |
| Mathematical object | Liouvillian derivative/Duhamel kernels | Pullback metric, two-form, connection, geodesics, transported frames |
| Reference dependence | Expansion about chosen local model | Coordinate-covariant comparison between separated models |
| Route between models | Iterative or straight parameter displacement | Minimum-energy physical geodesic |
| Truth observable | Target response used by recovery | Fixed endpoint 2D spectrum, explicitly path independent |
| Waiting time | Forward-model parameter | Experimental view defining identifiability/pathway geometry |
| Robustness | Accuracy and convergence of local recovery | Endpoint stability, waiting-time dependence, model mismatch, uncertainty bundles |
| Scaling question | Expansion order and response cost | Controls, path nodes, spectral pixels, and inverse identifiability |
| Scientific output | Efficient spectral transport/reconstruction | Geometry and uncertainty of model discovery |

The new manuscript may cite JCPIV_SAKE for the local expansion machinery, but
should not reproduce its principal derivation, validation suite, benchmark
story, or central figures. SAKE should appear as an enabling differentiable
forward engine, not as the new paper's claimed novelty.

## Core new theory

For controls \(\lambda\), a stationary-state/generator embedding

\[
\Phi(\lambda)=(r(\lambda),h(\lambda))
\]

induces

\[
G_{\mu\nu}
=\partial_\mu r\cdot\partial_\nu r
+\partial_\mu h\cdot\partial_\nu h,
\]

\[
F_{\mu\nu}
=\partial_\mu r\cdot\partial_\nu h
-\partial_\nu r\cdot\partial_\mu h,
\]

and the Levi-Civita connection \(\Gamma^\alpha_{\mu\nu}[G]\). These define a
physical geodesic and frame-transport law independently of arbitrary affine
interpolation in the chosen parameter coordinates.

The inverse and geometric stages must remain separate:

\[
Y_{\mathrm{truth}}
\longrightarrow q^\star
\longrightarrow \gamma_{p q^\star}
\longrightarrow \mathcal T_{\gamma}.
\]

The endpoint is inferred only from the observable:

\[
q^\star=\arg\min_q
\frac12\|W[\mathcal O(q)-Y_{\mathrm{truth}}]\|_2^2+R(q).
\]

After fixing \(q^\star\), the route is found from

\[
\gamma_{p q^\star}
=\arg\min_{\gamma(0)=p,\,\gamma(1)=q^\star}
\frac12\int_0^1
\dot\lambda^\mu G_{\mu\nu}(\lambda)\dot\lambda^\nu ds.
\]

## Two geometries that must not be conflated

The intrinsic physical metric is determined by the Liouvillian model:

\[
G^{\mathrm{phys}}_{\mu\nu}(q).
\]

The observation/identifiability metric depends on the experimental protocol:

\[
I_{\mu\nu}(q;t_2)
=J_{\mathcal O}(q;t_2)^T W^T W J_{\mathcal O}(q;t_2).
\]

If the recovered endpoint is unchanged, the intrinsic physical geodesic is
unchanged by \(t_2\). Observation-conditioned geodesics, local pathway frames,
and transported spectral subspaces may nevertheless depend strongly on
\(t_2\). This distinction is a central scientific result rather than a
technical detail.

## Minimum stand-alone contributions

The paper should not be submitted until it contains all of the following:

1. A model-independent construction of \(G\), \(F\), and \(\Gamma\) directly
   from differentiable Liouvillian stationary states and generator
   coordinates.
2. Coordinate-reparameterization tests showing that physical path length and
   transported inner products are invariant even when coordinate-space paths
   differ.
3. A validated boundary-value geodesic solver with autodiff gradients,
   refinement convergence, bounds, and conditioning diagnostics.
4. Parallel transport of actual local Liouville pathway/response frames, not
   only control-manifold tangent vectors.
5. Endpoint discovery from a fixed synthetic 2D spectrum followed by route
   discovery, demonstrating that the route cannot bias the recovered truth.
6. Waiting-time robustness: independent and joint multi-\(t_2\) endpoint
   recovery plus \(t_2\)-dependent pathway/observation geometry.
7. Model-adequacy tests using under-specified, correct, and over-specified
   candidate Liouvillians.
8. Endpoint uncertainty propagated into a bundle of geodesics and transported
   frames.
9. Scaling with control dimension, path resolution, and spectral-pixel count.
10. Generalized physical-observability modes from \(Iv=\chi Gv\), including a
    waiting-time design result or an independent curvature/holonomy prediction.

Items 4--8 and 10 are the decisive separation from an incremental methods
update.

## Validation hierarchy

### Level 1: analytic optical Bloch geometry

- Closed-form stationary state.
- Symbolic, finite-difference, and Liouvillian-autodiff agreement.
- Metric positivity, two-form antisymmetry, and connection symmetry.
- Known or independently verified geodesics and transport invariants.

### Level 2: exciton dimer transfer

- New-model robustness with no optical-Bloch-specific code paths.
- Three or more controls, including coherent coupling, detailed-balance
  population transfer, and collective/local dephasing.
- Straight versus physical-geodesic comparisons.
- Genuine Liouville pathway-frame transport.

### Level 3: inverse spectral discovery

- Generate a hidden endpoint \(q_{\mathrm{true}}\).
- Produce fixed 2D truth spectra.
- Recover \(q^\star\) from multiple starts without geometric path penalties.
- Compute the geodesic only after endpoint recovery.
- Report both spectral error and control-space recovery error.

### Level 4: robustness and discrimination

- Noise ensembles and endpoint covariance.
- Independent and joint multi-waiting-time fits.
- Held-out \(t_2\) prediction.
- Nested-model discrimination and structured residual analysis.
- Geodesic/pathway-frame uncertainty bundles.

## Proposed principal figures

1. **Concept and separation of stages.** Fixed truth spectrum to endpoint
   discovery, followed by physical route discovery and covariant transport.
2. **Optical Bloch analytic validation.** Agreement of all geometry backends
   and coordinate-reparameterization invariance.
3. **Dimer physical manifold.** Straight and geodesic paths with metric and
   transport diagnostics.
4. **Endpoint recovery.** Hidden truth, recovered endpoint, residual spectrum,
   convergence, and identifiability singular values.
5. **Waiting-time robustness.** \(q^\star(t_2)\), generalized observability
   eigenvalues \(\chi_a(t_2)\), and transported-frame/pathway changes at fixed
   truth.
6. **Model discrimination.** Underfit/correct/overfit residual structure and
   held-out multi-\(t_2\) performance.
7. **Uncertainty geometry.** Endpoint confidence region and corresponding
   geodesic bundle.
8. **Scaling.** Runtime and memory versus controls, path nodes, and spectral
   pixels, including finite-difference versus autodiff path optimization.

## Decision gates

The work is a stand-alone paper only if the following questions have affirmative
answers:

- Does the covariant construction produce a measurable advantage or new
  invariant unavailable from straight parameter interpolation?
- Does actual pathway-frame transport yield information beyond tangent-frame
  transport?
- Does waiting time reveal changing observation/pathway geometry at a stable
  recovered endpoint?
- Can the inversion diagnose missing and unnecessary model terms?
- Can uncertainty in the endpoint be propagated meaningfully into route and
  transport uncertainty?
- Does the generalized \((I,G)\) spectrum reveal observable/dark physical modes
  or select waiting times in a way ordinary parameter fitting cannot?

If the results are limited to replacing straight interpolation by a slightly
lower-energy path, the work remains an incremental extension. The stand-alone
scientific contribution is the geometry, identifiability, and robustness of
model discovery from nonlinear spectra.

## Immediate development sequence

1. Replace finite-difference path optimization with a JAX-gradient objective.
2. Implement a differentiable endpoint observable for the dimer 2D spectrum.
3. Recover a fixed synthetic endpoint from several initial guesses.
4. Map control tangent directions into local Liouville pathway frames and
   transport those frames along the recovered geodesic.
5. Run the waiting-time and model-discrimination matrices.
6. Compute generalized \((I,G)\) observability modes and optimize a multi-time
   acquisition design.
7. Add uncertainty propagation and three-axis scaling benchmarks.

No changes from this program should be merged into the under-review
JCPIV_SAKE publication branch. Shared machinery should be imported or copied
only after that manuscript's archival state is fixed and an explicit
integration decision is made.
