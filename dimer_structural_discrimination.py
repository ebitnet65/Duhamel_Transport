"""Sensitivity and hidden-channel tools for the exciton-dimer spectrum.

The original three-parameter observable remains the regression benchmark.
This module adds an optically dark terminal sink.  Optical response is
computed on the four-state active block with the exact trace-decreasing
generator induced by the five-state trace-preserving trap model.  The same
generator is used in all coherence, waiting, and detection intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dimer_spectral_observable import DimerSpectrumConfig, _require_jax, _sector_fields
from itertools import product


def build_structural_observable(config: DimerSpectrumConfig, mechanism: str = "A"):
    """Build complex spectrum for A (dimer), B (P1 dephasing), or C (trap).

    Parameters are ``[J12, k12, gamma_phi, extra_rate]``.  For A the fourth
    entry is ignored; B and C interpret it as ``gamma_d`` and ``k_d``.
    The trap is terminal, dark, and has no dipole or return pathway.
    """
    mechanism = mechanism.upper()
    if mechanism not in {"A", "B", "C"}:
        raise ValueError("mechanism must be A, B, or C")
    _jax, jnp, jsp = _require_jax()
    w1 = jnp.asarray(config.omega1_grid, dtype=jnp.float64)
    w3 = jnp.asarray(config.omega3_grid, dtype=jnp.float64)
    dim, n = 4, 16
    ident = jnp.eye(dim, dtype=jnp.complex128)
    ident_l = jnp.eye(n, dtype=jnp.complex128)

    def op(i, j):
        return jnp.zeros((dim, dim), dtype=jnp.complex128).at[i, j].set(1.)

    def diss(c):
        cdc = c.conj().T @ c
        return jnp.kron(c.conj(), c) - .5*jnp.kron(ident, cdc) - .5*jnp.kron(cdc.T, ident)

    p = tuple(op(i, i) for i in range(dim))
    mup = (config.mu_1*op(1, 0) + config.mu_2*op(2, 0)
           + config.mu_esa_scale*config.mu_2*op(3, 1)
           + config.mu_esa_scale*config.mu_1*op(3, 2))
    mum, mu = mup.conj().T, mup + mup.conj().T
    rho0 = p[0].reshape((-1,), order="F")
    z = jnp.diag(jnp.asarray([-2., 0., 0., 2.]))

    def left(a): return jnp.kron(ident, a)
    def right(a): return jnp.kron(a.T, ident)
    interactions = {(side, field): (left(comp) if side == "L" else -right(comp))
                    for side in ("L", "R")
                    for field, comp in (("+", mum), ("-", mup))}
    pathways = tuple((s.upper(), sides, _sector_fields(s)) for s in config.sectors
                     for sides in product(("L", "R"), repeat=3))

    def generator(q):
        coupling, transfer, dephasing, extra = q
        e1, e2 = config.mean_site_energy-.5*config.detuning, config.mean_site_energy+.5*config.detuning
        h = e1*p[1] + e2*p[2] + (e1+e2+config.anharmonicity)*p[3] + coupling*(op(1,2)+op(2,1))
        l = -1j*(jnp.kron(ident,h)-jnp.kron(h.T,ident))
        l += config.gamma_1*(diss(op(0,1))+diss(op(2,3)))
        l += config.gamma_2*(diss(op(0,2))+diss(op(1,3)))
        l += transfer*jnp.exp(-.5*config.beta*config.detuning)*diss(op(2,1))
        l += transfer*jnp.exp(+.5*config.beta*config.detuning)*diss(op(1,2))
        l += dephasing*diss(z)
        if mechanism == "B":
            l += extra*diss(p[1])
        elif mechanism == "C":
            # Exact active-block generator for L=sqrt(kd)|d><e1|.
            # P1=sigma_11; this acts in t1, t2, and t3 propagators.
            l += -.5*extra*(jnp.kron(ident,p[1])+jnp.kron(p[1].T,ident))
        return l

    def resolvent(l, omega):
        return -jnp.linalg.solve(l+(1j*omega-config.eta)*ident_l, ident_l)

    def detect(state):
        return jnp.trace(mu @ state.reshape((dim,dim), order="F"))

    def observable(q, t2):
        l = generator(jnp.asarray(q, dtype=jnp.float64))
        wait = jsp.linalg.expm(l*jnp.asarray(t2,dtype=jnp.float64))
        out = jnp.zeros((w3.size,w1.size), dtype=jnp.complex128)
        for sector,sides,fields in pathways:
            j1,j2,j3 = (interactions[(sides[i],fields[i])] for i in range(3))
            sign = 1. if sector == "RP" else -1.
            for i in range(w1.size):
                state = resolvent(l, sign*w1[i]) @ j1 @ rho0
                state = j3 @ wait @ j2 @ state
                for k in range(w3.size):
                    out = out.at[k,i].add(detect(resolvent(l,-w3[k]) @ state))
        return out
    return observable


def full_trap_liouvillian(k_d: float, *, dimension: int = 5) -> np.ndarray:
    """Trace-preserving generator for an isolated |e1> -> |d> trap jump."""
    if dimension < 3 or k_d < 0:
        raise ValueError("dimension must be >=3 and k_d nonnegative")
    e = np.zeros((dimension,dimension),complex)
    e[-1,1] = np.sqrt(k_d)
    ident = np.eye(dimension,dtype=complex)
    cdc=e.conj().T@e
    return np.kron(e.conj(),e)-.5*np.kron(ident,cdc)-.5*np.kron(cdc.T,ident)


def full_pure_dephasing_liouvillian(gamma_d: float, *, dimension: int = 5) -> np.ndarray:
    """Generator for the distinct local jump sqrt(gamma_d)|e1><e1|."""
    if dimension < 3 or gamma_d < 0:
        raise ValueError("dimension must be >=3 and gamma_d nonnegative")
    p=np.zeros((dimension,dimension),complex); p[1,1]=1.
    ident=np.eye(dimension,dtype=complex); cdc=p
    return gamma_d*(np.kron(p.conj(),p)-.5*np.kron(ident,cdc)-.5*np.kron(cdc.T,ident))


def finite_difference_jacobian(observable, q, t2, steps=None):
    q=np.asarray(q,float)
    if steps is None: steps=np.maximum(1e-7, np.abs(q)*1e-5)
    cols=[]
    for i,h in enumerate(steps):
        delta=np.zeros_like(q); delta[i]=h
        cols.append((np.asarray(observable(q+delta,t2))-np.asarray(observable(q-delta,t2)))/(2*h))
    return np.stack(cols,axis=-1)


def whitened_diagnostics(jacobian, sigma: float = 1.0):
    """Real/imaginary stacked SVD information and parameter correlations."""
    j=np.asarray(jacobian)
    flat=j.reshape((-1,j.shape[-1]))
    if np.iscomplexobj(flat): flat=np.vstack((flat.real,flat.imag))
    jw=flat/float(sigma)
    u,s,vt=np.linalg.svd(jw,full_matrices=False)
    tol=max(jw.shape)*np.finfo(float).eps*(s[0] if s.size else 0.)
    rank=int(np.sum(s>tol))
    info=jw.T@jw
    cov=np.linalg.pinv(info,rcond=1e-12)
    scale=np.sqrt(np.maximum(np.diag(cov),0))
    corr=cov/np.maximum(np.outer(scale,scale),np.finfo(float).tiny)
    return {"information":info,"singular_values":s,"rank":rank,
            "condition":float(s[0]/s[rank-1]) if rank else float("inf"),
            "correlation":corr,"right_singular_vectors":vt}


__all__=["build_structural_observable","full_trap_liouvillian",
         "full_pure_dephasing_liouvillian","finite_difference_jacobian",
         "whitened_diagnostics"]
