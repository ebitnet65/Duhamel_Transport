"""
Liouville_qutip_adapter.py

Optional QuTiP bridge for the general Liouville transport core.

QuTiP is intentionally not a hard dependency. This module converts QuTiP-style
`Qobj` definitions at the boundary and returns a `LiouvilleModel` using the core
matrix protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from Liouville_transport_general import LiouvilleModel

Params = Mapping[str, float]


def _to_dense_array(obj: Any) -> np.ndarray:
    """Convert a QuTiP Qobj or array-like object to a dense complex array."""

    if hasattr(obj, "full"):
        return np.asarray(obj.full(), dtype=complex)
    return np.asarray(obj, dtype=complex)


def qutip_model_from_hamiltonian(
    *,
    dimension: int,
    hamiltonian: Callable[[Params], Any],
    collapse_ops: Callable[[Params], Sequence[Any]],
    initial_state: Callable[[Params], Any],
    transition_dipole: Callable[[Params], Any],
    dipole_components: Callable[[Params], tuple[Any, Any]] | None = None,
) -> LiouvilleModel:
    """Create a `LiouvilleModel` from QuTiP-style Qobj callables.

    The callables may return actual QuTiP `Qobj` instances or ordinary array-like
    matrices/vectors. This keeps tests portable when QuTiP is not installed.
    """

    def hamiltonian_array(params: Params) -> np.ndarray:
        return _to_dense_array(hamiltonian(params))

    def collapse_arrays(params: Params) -> tuple[np.ndarray, ...]:
        return tuple(_to_dense_array(collapse) for collapse in collapse_ops(params))

    def initial_state_array(params: Params) -> np.ndarray:
        rho = _to_dense_array(initial_state(params))
        return rho.reshape(-1, order="F")

    def transition_dipole_array(params: Params) -> np.ndarray:
        return _to_dense_array(transition_dipole(params))

    def dipole_component_arrays(params: Params) -> tuple[np.ndarray, np.ndarray]:
        if dipole_components is None:
            raise RuntimeError("dipole_component_arrays called without dipole_components")
        mu_plus, mu_minus = dipole_components(params)
        return _to_dense_array(mu_plus), _to_dense_array(mu_minus)

    return LiouvilleModel.from_hamiltonian(
        dimension=dimension,
        hamiltonian=hamiltonian_array,
        collapse_ops=collapse_arrays,
        initial_state=initial_state_array,
        transition_dipole=transition_dipole_array,
        dipole_components=dipole_component_arrays if dipole_components is not None else None,
    )


__all__ = ["qutip_model_from_hamiltonian"]
