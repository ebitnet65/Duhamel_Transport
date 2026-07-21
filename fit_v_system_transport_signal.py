#!/usr/bin/env python3
"""
fit_v_system_transport_signal.py

Proof-of-concept inverse problem for pathway transport.

The script:

1. Builds the V-system reference model.
2. Generates a reference 2D third-order signal.
3. Draws a random target lambda-vector and computes the exact target signal.
4. Builds a second-order transport expansion T(lambda) on the same 2D grid.
5. Fits lambda by minimizing the residual between the target signal and the
   signal transported from the lambda=0 pathway basis.

This is intentionally finite-difference based. The autodiff/Duhamel backend can
replace the transport-derivative construction later without changing the
workflow tested here.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os

import numpy as np
from scipy.optimize import minimize

import Liouville_transport_general as general
from Liouville_transport_expansion import TransportDerivativeExpansion
from test_v_system_general import v_system_model, v_system_parameter_map

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-duhamel-transport")
import matplotlib.pyplot as plt


def trace_signal(model: general.LiouvilleModel, params: general.Params, state: np.ndarray) -> complex:
    """Contract a Liouville-space state with the transition dipole observable."""

    rho = general.mat(state, model.dimension)
    return np.trace(model.mu(params) @ rho)


def pathway_signal_grid(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    pathways: list[general.CanonicalPathway],
    *,
    tau2: float,
    eta: float,
) -> np.ndarray:
    """Return flattened 2D signals with one column per pathway."""

    params = parameter_map.params(lam)
    columns = []
    for item in pathways:
        values = np.empty((len(omega3_values), len(omega1_values)), dtype=complex)
        for i3, omega3 in enumerate(omega3_values):
            for i1, omega1 in enumerate(omega1_values):
                state = general.pathway_state_point(
                    model,
                    parameter_map,
                    lam,
                    omega1,
                    omega3,
                    item.pathway,
                    item.sector,
                    stage="after_G3",
                    tau2=tau2,
                    eta=eta,
                )
                values[i3, i1] = trace_signal(model, params, state)
        columns.append(values.reshape(-1))
    return np.column_stack(columns)


def pathway_state_grid_matrix(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    pathways: list[general.CanonicalPathway],
    *,
    tau2: float,
    eta: float,
) -> np.ndarray:
    """Return stacked final-stage pathway states on a full 2D frequency grid."""

    weights = np.sqrt(np.outer(np.gradient(omega3_values), np.gradient(omega1_values)))
    columns = []
    for item in pathways:
        states = []
        for i3, omega3 in enumerate(omega3_values):
            for i1, omega1 in enumerate(omega1_values):
                state = general.pathway_state_point(
                    model,
                    parameter_map,
                    lam,
                    omega1,
                    omega3,
                    item.pathway,
                    item.sector,
                    stage="after_G3",
                    tau2=tau2,
                    eta=eta,
                )
                states.append(weights[i3, i1] * state)
        columns.append(np.asarray(states).reshape(-1))
    return np.column_stack(columns)


def commutator_signal_grid(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    lam: np.ndarray,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    *,
    tau2: float,
    eta: float,
) -> np.ndarray:
    """Return the direct full-commutator 2D third-order response."""

    params = parameter_map.params(lam)
    L = model.L(params)
    rho0 = model.rho0(params)
    mu = model.mu(params)
    J = general.commutator_super(mu)
    U2 = model.U(L, tau2)

    signal = np.empty((len(omega3_values), len(omega1_values)), dtype=complex)
    for i3, omega3 in enumerate(omega3_values):
        G3 = model.G(L, omega3, eta)
        for i1, omega1 in enumerate(omega1_values):
            G1 = model.G(L, omega1, eta)
            state = G3 @ J @ U2 @ J @ G1 @ J @ rho0
            signal[i3, i1] = trace_signal(model, params, state)
    return signal.reshape(-1)


def project_transport_derivative(
    P0: np.ndarray,
    derivative: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project derivative columns onto the reference pathway-state basis."""

    coefficients = np.linalg.pinv(P0, rcond=rcond) @ derivative
    reconstructed = P0 @ coefficients
    residuals = np.array([
        np.linalg.norm(derivative[:, index] - reconstructed[:, index])
        / max(np.linalg.norm(derivative[:, index]), 1e-14)
        for index in range(derivative.shape[1])
    ])
    return coefficients.T, residuals


def finite_difference_transport_expansion_2d(
    model: general.LiouvilleModel,
    parameter_map: general.ParameterMap,
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    pathways: list[general.CanonicalPathway],
    *,
    tau2: float,
    eta: float,
    step: float,
    rcond: float,
) -> TransportDerivativeExpansion:
    """Build a second-order T(lambda) expansion on a full 2D grid."""

    n_controls = parameter_map.control_dimension
    zero = parameter_map.zero()
    P0 = pathway_state_grid_matrix(
        model,
        parameter_map,
        zero,
        omega1_values,
        omega3_values,
        pathways,
        tau2=tau2,
        eta=eta,
    )

    T0, residual0 = project_transport_derivative(P0, P0, rcond=rcond)
    first = {}
    second = {}
    first_residuals = {}
    second_residuals = {}

    for mu in range(n_controls):
        direction = np.zeros(n_controls)
        direction[mu] = step
        Pp = pathway_state_grid_matrix(
            model,
            parameter_map,
            zero + direction,
            omega1_values,
            omega3_values,
            pathways,
            tau2=tau2,
            eta=eta,
        )
        Pm = pathway_state_grid_matrix(
            model,
            parameter_map,
            zero - direction,
            omega1_values,
            omega3_values,
            pathways,
            tau2=tau2,
            eta=eta,
        )
        first[mu], first_residuals[mu] = project_transport_derivative(
            P0,
            (Pp - Pm) / (2.0 * step),
            rcond=rcond,
        )
        second[(mu, mu)], second_residuals[(mu, mu)] = project_transport_derivative(
            P0,
            (Pp - 2.0 * P0 + Pm) / (step * step),
            rcond=rcond,
        )

    for mu in range(n_controls):
        for nu in range(mu + 1, n_controls):
            emu = np.zeros(n_controls)
            enu = np.zeros(n_controls)
            emu[mu] = step
            enu[nu] = step
            matrices = []
            for lam in (
                zero + emu + enu,
                zero + emu - enu,
                zero - emu + enu,
                zero - emu - enu,
            ):
                matrices.append(
                    pathway_state_grid_matrix(
                        model,
                        parameter_map,
                        lam,
                        omega1_values,
                        omega3_values,
                        pathways,
                        tau2=tau2,
                        eta=eta,
                    )
                )
            mixed = (matrices[0] - matrices[1] - matrices[2] + matrices[3]) / (
                4.0 * step * step
            )
            second[(mu, nu)], second_residuals[(mu, nu)] = project_transport_derivative(
                P0,
                mixed,
                rcond=rcond,
            )

    return TransportDerivativeExpansion(
        T0=T0,
        first=first,
        second=second,
        residual0=residual0,
        first_residuals=first_residuals,
        second_residuals=second_residuals,
        labels=tuple(item.label for item in pathways),
        stage="after_G3",
    )


def transported_signal(
    reference_pathway_signals: np.ndarray,
    expansion: TransportDerivativeExpansion,
    lam: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Transport the lambda=0 pathway signals and sum over pathways."""

    T = expansion.evaluate(lam, order=order)
    pathway_weights = T.T @ np.ones(T.shape[0], dtype=complex)
    return reference_pathway_signals @ pathway_weights


def relative_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - actual) / max(np.linalg.norm(actual), 1e-14))


def plot_signal_panel(
    omega1_values: np.ndarray,
    omega3_values: np.ndarray,
    signal: np.ndarray,
    title: str,
    output_file: Path,
    *,
    component: str = "real",
) -> None:
    if component == "real":
        values = np.real(signal)
        label = r"$\mathrm{Re}\,S$"
        cmap = "RdBu_r"
    elif component == "imag":
        values = np.imag(signal)
        label = r"$\mathrm{Im}\,S$"
        cmap = "RdBu_r"
    elif component == "abs":
        values = np.abs(signal)
        label = r"$|S|$"
        cmap = "magma"
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image_values = values.reshape(len(omega3_values), len(omega1_values))
    if component in ("real", "imag"):
        vmax = float(np.max(np.abs(image_values)))
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.max(image_values))
    image = ax.imshow(
        image_values,
        origin="lower",
        extent=(
            float(omega1_values[0]),
            float(omega1_values[-1]),
            float(omega3_values[0]),
            float(omega3_values[-1]),
        ),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\omega_1$")
    ax.set_ylabel(r"$\omega_3$")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit V-system lambda controls from a transported 2D signal."
    )
    parser.add_argument("--n-omega", type=int, default=40)
    parser.add_argument("--omega-min", type=float, default=0.82)
    parser.add_argument("--omega-max", type=float, default=1.34)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2e-3)
    parser.add_argument("--fd-step", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-radius", type=float, default=0.35)
    parser.add_argument("--fit-bound", type=float, default=0.8)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=("RP", "NRP"),
        choices=("RP", "NRP", "DQC"),
        help="Phase-matched sectors included in the fitted signal.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("transport_fit_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = v_system_model()
    parameter_map = v_system_parameter_map(
        general.ControlSpec("J_bd", reference_value=0.0, scale=0.02),
        general.ControlSpec("k_bd", reference_value=0.006, scale=0.003),
    )
    omega1_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega3_values = np.linspace(args.omega_min, args.omega_max, args.n_omega)

    sectors = tuple(sector.upper() for sector in args.sectors)
    candidates = general.canonical_pathways(sectors)
    pathways, _ = general.independent_canonical_pathways(
        model,
        parameter_map,
        omega1_values,
        candidates,
        stage="after_G3",
        tau2=args.tau2,
        eta=args.eta,
    )

    rng = np.random.default_rng(args.seed)
    target_lam = rng.uniform(-args.target_radius, args.target_radius, size=2)

    reference_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
    )
    target_pathway_signals = pathway_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
    )
    reference_signal = np.sum(reference_pathway_signals, axis=1)
    target_signal = np.sum(target_pathway_signals, axis=1)
    reference_commutator_signal = commutator_signal_grid(
        model,
        parameter_map,
        parameter_map.zero(),
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )
    target_commutator_signal = commutator_signal_grid(
        model,
        parameter_map,
        target_lam,
        omega1_values,
        omega3_values,
        tau2=args.tau2,
        eta=args.eta,
    )

    expansion = finite_difference_transport_expansion_2d(
        model,
        parameter_map,
        omega1_values,
        omega3_values,
        pathways,
        tau2=args.tau2,
        eta=args.eta,
        step=args.fd_step,
        rcond=1e-12,
    )

    def objective(lam: np.ndarray) -> float:
        predicted = transported_signal(
            reference_pathway_signals,
            expansion,
            lam,
            order=args.order,
        )
        return relative_error(target_signal, predicted) ** 2

    result = minimize(
        objective,
        x0=np.zeros(parameter_map.control_dimension),
        method="Powell",
        bounds=[(-args.fit_bound, args.fit_bound)] * parameter_map.control_dimension,
        options={"xtol": 1e-8, "ftol": 1e-12, "maxiter": 250},
    )
    fit_lam = result.x
    fitted_signal = transported_signal(
        reference_pathway_signals,
        expansion,
        fit_lam,
        order=args.order,
    )
    target_transport_signal = transported_signal(
        reference_pathway_signals,
        expansion,
        target_lam,
        order=args.order,
    )

    for component in ("real", "imag", "abs"):
        plot_signal_panel(
            omega1_values,
            omega3_values,
            reference_commutator_signal,
            "Reference full-commutator signal",
            args.output_dir / f"reference_signal_{component}.png",
            component=component,
        )
        plot_signal_panel(
            omega1_values,
            omega3_values,
            target_commutator_signal,
            "Exact target full-commutator signal",
            args.output_dir / f"target_signal_{component}.png",
            component=component,
        )
        plot_signal_panel(
            omega1_values,
            omega3_values,
            reference_signal,
            f"Reference pathway-sum signal ({'+'.join(sectors)})",
            args.output_dir / f"reference_pathway_sum_signal_{component}.png",
            component=component,
        )
        plot_signal_panel(
            omega1_values,
            omega3_values,
            target_signal,
            f"Exact target pathway-sum signal ({'+'.join(sectors)})",
            args.output_dir / f"target_pathway_sum_signal_{component}.png",
            component=component,
        )
        plot_signal_panel(
            omega1_values,
            omega3_values,
            fitted_signal,
            f"Best transported pathway fit ({'+'.join(sectors)})",
            args.output_dir / f"fit_pathway_sum_signal_{component}.png",
            component=component,
        )
        plot_signal_panel(
            omega1_values,
            omega3_values,
            fitted_signal - target_signal,
            f"Transported pathway-fit residual ({'+'.join(sectors)})",
            args.output_dir / f"fit_pathway_sum_residual_{component}.png",
            component=component,
        )

    np.savez_compressed(
        args.output_dir / "fit_v_system_transport_signal.npz",
        omega1_values=omega1_values,
        omega3_values=omega3_values,
        target_lam=target_lam,
        fit_lam=fit_lam,
        reference_signal=reference_signal,
        target_signal=target_signal,
        reference_commutator_signal=reference_commutator_signal,
        target_commutator_signal=target_commutator_signal,
        fitted_signal=fitted_signal,
        target_transport_signal=target_transport_signal,
        labels=np.asarray(expansion.labels, dtype=str),
        sectors=np.asarray(sectors, dtype=str),
        T0=expansion.T0,
        T_first=np.asarray([expansion.first[index] for index in sorted(expansion.first)]),
        T_second_keys=np.asarray(list(expansion.second), dtype=int),
        T_second=np.asarray([expansion.second[key] for key in expansion.second]),
    )

    summary_file = args.output_dir / "fit_summary.txt"
    with summary_file.open("w", encoding="utf-8") as handle:
        handle.write("V-system transported-signal fit\n")
        handle.write("=" * 72 + "\n\n")
        handle.write(f"sectors: {list(sectors)}\n")
        handle.write(f"controls: {[control.name for control in parameter_map.controls]}\n")
        handle.write(f"target lambda: {np.array2string(target_lam, precision=8)}\n")
        handle.write(f"fit lambda   : {np.array2string(fit_lam, precision=8)}\n")
        handle.write(f"lambda error : {np.linalg.norm(fit_lam - target_lam):.12e}\n")
        handle.write(f"objective    : {result.fun:.12e}\n")
        handle.write(f"fit success  : {result.success}\n")
        handle.write(f"message      : {result.message}\n\n")
        handle.write(
            "reference-to-target exact relative change: "
            f"{relative_error(target_signal, reference_signal):.12e}\n"
        )
        handle.write(
            "transport expansion error at target lambda: "
            f"{relative_error(target_signal, target_transport_signal):.12e}\n"
        )
        handle.write(
            "transport expansion error at fitted lambda: "
            f"{relative_error(target_signal, fitted_signal):.12e}\n\n"
        )
        handle.write("pathway labels\n")
        handle.write("-" * 72 + "\n")
        for index, label in enumerate(expansion.labels):
            handle.write(f"{index:2d}: {label}\n")

    print("V-system transported-signal fit")
    print("=" * 72)
    print(f"target lambda: {np.array2string(target_lam, precision=8)}")
    print(f"fit lambda   : {np.array2string(fit_lam, precision=8)}")
    print(f"lambda error : {np.linalg.norm(fit_lam - target_lam):.6e}")
    print(f"target expansion error: {relative_error(target_signal, target_transport_signal):.6e}")
    print(f"fit expansion error   : {relative_error(target_signal, fitted_signal):.6e}")
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
