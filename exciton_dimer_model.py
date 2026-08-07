"""
exciton_dimer_model.py

Four-level site-basis exciton dimer model.

Basis:

    |0> = |00>
    |1> = |10>
    |2> = |01>
    |3> = |11>

The transition dipole is defined in the site basis. Ground-to-one-exciton
transitions are always included; one-exciton-to-double-exciton transitions are
controlled by `mu_esa_scale`.
"""

from __future__ import annotations

import numpy as np

import Liouville_transport_general as general


def ket(index: int, dim: int = 4) -> np.ndarray:
    out = np.zeros((dim, 1), dtype=complex)
    out[index, 0] = 1.0
    return out


def projector(index: int, dim: int = 4) -> np.ndarray:
    state = ket(index, dim)
    return state @ state.conj().T


def op(i: int, j: int, dim: int = 4) -> np.ndarray:
    return ket(i, dim) @ ket(j, dim).conj().T


def detailed_balance_site_transfer_rates(params: general.Params) -> tuple[float, float]:
    """Return incoherent site-transfer rates for |10> -> |01> and reverse."""

    delta = params["omega_2"] - params["omega_1"]
    if "beta_12" in params:
        beta = params["beta_12"]
    else:
        temperature = params["temperature"]
        beta = np.inf if temperature <= 0 else 1.0 / temperature
    rate = params["k_12"]
    if np.isinf(beta):
        if delta > 0:
            return 0.0, rate
        if delta < 0:
            return rate, 0.0
        return rate, rate
    return (
        rate * np.exp(-0.5 * beta * delta),
        rate * np.exp(+0.5 * beta * delta),
    )


def exciton_dimer_hamiltonian(params: general.Params) -> np.ndarray:
    """Return the Hilbert-space Hamiltonian in the site-product basis."""

    return (
        params["omega_1"] * projector(1)
        + params["omega_2"] * projector(2)
        + (params["omega_1"] + params["omega_2"] + params["anharmonicity"]) * projector(3)
        + params["J_12"] * (op(1, 2) + op(2, 1))
    )


def exciton_dimer_lindblad_ops(params: general.Params) -> tuple[np.ndarray, ...]:
    """Return Lindblad collapse operators for the dimer model."""

    collapse_ops = []
    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    gamma_phi_1 = params["gamma_phi_1"]
    gamma_phi_2 = params["gamma_phi_2"]
    k_1_to_2, k_2_to_1 = detailed_balance_site_transfer_rates(params)

    if gamma_1 > 0:
        collapse_ops.append(np.sqrt(gamma_1) * op(0, 1))
        collapse_ops.append(np.sqrt(gamma_1) * op(2, 3))
    if gamma_2 > 0:
        collapse_ops.append(np.sqrt(gamma_2) * op(0, 2))
        collapse_ops.append(np.sqrt(gamma_2) * op(1, 3))
    if gamma_phi_1 > 0:
        collapse_ops.append(np.sqrt(gamma_phi_1) * (projector(1) + projector(3)))
    if gamma_phi_2 > 0:
        collapse_ops.append(np.sqrt(gamma_phi_2) * (projector(2) + projector(3)))
    if k_1_to_2 > 0:
        collapse_ops.append(np.sqrt(k_1_to_2) * op(2, 1))
    if k_2_to_1 > 0:
        collapse_ops.append(np.sqrt(k_2_to_1) * op(1, 2))
    return tuple(collapse_ops)


def exciton_dimer_mu_components(params: general.Params) -> tuple[np.ndarray, np.ndarray]:
    """Return raising/lowering dipole components including optional ESA."""

    mu_1 = params["mu_1"]
    mu_2 = params["mu_2"]
    esa_scale = params["mu_esa_scale"]
    mu_plus = (
        mu_1 * op(1, 0)
        + mu_2 * op(2, 0)
        + esa_scale * mu_2 * op(3, 1)
        + esa_scale * mu_1 * op(3, 2)
    )
    mu_minus = mu_plus.conj().T
    return mu_plus, mu_minus


def exciton_dimer_model() -> general.LiouvilleModel:
    return general.LiouvilleModel.from_hamiltonian(
        dimension=4,
        hamiltonian=exciton_dimer_hamiltonian,
        collapse_ops=exciton_dimer_lindblad_ops,
        initial_state=lambda _params: general.vec(projector(0)),
        transition_dipole=lambda params: sum(exciton_dimer_mu_components(params)),
        dipole_components=exciton_dimer_mu_components,
    )


def exciton_dimer_parameter_map(*controls: general.ControlSpec) -> general.ParameterMap:
    if not controls:
        controls = (general.ControlSpec("J_12", reference_value=0.0),)
    return general.ParameterMap(
        reference_params={
            "omega_1": 1.0,
            "omega_2": 1.18,
            "anharmonicity": -0.06,
            "J_12": 0.0,
            "gamma_1": 0.03,
            "gamma_2": 0.004,
            "gamma_phi_1": 0.02,
            "gamma_phi_2": 0.012,
            "k_12": 0.006,
            "temperature": 0.4,
            "mu_1": 1.0,
            "mu_2": 1.0,
            "mu_esa_scale": 1.0,
        },
        controls=tuple(controls),
    )


def one_exciton_summary(params: general.Params) -> list[dict[str, float]]:
    """Return one-exciton energies and oscillator strengths."""

    h_one = exciton_dimer_hamiltonian(params)[1:3, 1:3]
    energies, vectors = np.linalg.eigh(h_one)
    site_dipole = np.array([params["mu_1"], params["mu_2"]], dtype=complex)
    out = []
    for index, energy in enumerate(energies):
        vector = np.real_if_close(vectors[:, index])
        amplitude = np.vdot(site_dipole, vector)
        out.append(
            {
                "energy": float(energy),
                "site10_coeff": float(vector[0]),
                "site01_coeff": float(vector[1]),
                "oscillator_strength": float(abs(amplitude) ** 2),
            }
        )
    return out


def excited_state_absorption_frequencies(params: general.Params) -> list[float]:
    """Return transition frequencies from one-exciton states to |11>."""

    double_energy = params["omega_1"] + params["omega_2"] + params["anharmonicity"]
    return [double_energy - item["energy"] for item in one_exciton_summary(params)]


__all__ = [
    "detailed_balance_site_transfer_rates",
    "excited_state_absorption_frequencies",
    "exciton_dimer_hamiltonian",
    "exciton_dimer_lindblad_ops",
    "exciton_dimer_model",
    "exciton_dimer_mu_components",
    "exciton_dimer_parameter_map",
    "ket",
    "one_exciton_summary",
    "op",
    "projector",
]
