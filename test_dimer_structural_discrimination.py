import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from dimer_spectral_observable import DimerSpectrumConfig, build_dimer_spectral_observable
from dimer_structural_discrimination import (
    build_structural_observable, finite_difference_jacobian,
    full_pure_dephasing_liouvillian, full_trap_liouvillian,
)


def test_baseline_extended_factory_is_unchanged():
    grid=np.array([.98,1.12])
    cfg=DimerSpectrumConfig(grid,grid,eta=.04)
    q=np.array([.025,.014,.022])
    base=np.asarray(build_dimer_spectral_observable(cfg)(q,.7))
    extended=np.asarray(build_structural_observable(cfg,"A")([*q,0.],.7))
    np.testing.assert_allclose(extended,base,rtol=1e-12,atol=1e-12)
    no_trap=np.asarray(build_structural_observable(cfg,"C")([*q,0.],.7))
    np.testing.assert_allclose(no_trap,base,rtol=1e-12,atol=1e-12)


def test_structural_spectral_ad_matches_centered_differences():
    grid=np.array([.96,1.08,1.20])
    cfg=DimerSpectrumConfig(grid,grid,eta=.04)
    fn=build_structural_observable(cfg,"C")
    q=jnp.array([.025,.014,.022,.008])
    jac=np.asarray(jax.jacfwd(lambda x: fn(x,.8))(q))
    fd=finite_difference_jacobian(fn,np.asarray(q),.8)
    np.testing.assert_allclose(jac,fd,rtol=4e-5,atol=3e-6)


def test_trap_generator_trace_and_population_accounting():
    kd=.17
    l=full_trap_liouvillian(kd)
    rho=np.zeros((5,5),complex); rho[1,1]=.7; rho[1,2]=.2+.1j; rho[2,1]=rho[1,2].conjugate()
    dr=(l@rho.reshape(-1,order="F")).reshape((5,5),order="F")
    np.testing.assert_allclose(np.trace(dr),0,atol=1e-14)
    np.testing.assert_allclose(dr[1,1],-kd*rho[1,1])
    np.testing.assert_allclose(dr[4,4],kd*rho[1,1])
    np.testing.assert_allclose(dr[1,2],-.5*kd*rho[1,2])
    active=dr[:4,:4]
    np.testing.assert_allclose(np.trace(active),-kd*rho[1,1])
    p1=np.zeros((4,4),complex); p1[1,1]=1
    expected=-.5*kd*(p1@rho[:4,:4]+rho[:4,:4]@p1)
    np.testing.assert_allclose(active,expected,atol=1e-14)


def test_pure_dephasing_does_not_remove_population():
    gamma=.23; l=full_pure_dephasing_liouvillian(gamma)
    rho=np.zeros((5,5),complex); rho[1,1]=.8; rho[1,2]=.3j; rho[2,1]=-.3j
    dr=(l@rho.reshape(-1,order="F")).reshape((5,5),order="F")
    np.testing.assert_allclose(dr[1,1],0,atol=1e-14)
    np.testing.assert_allclose(dr[4,4],0,atol=1e-14)
    np.testing.assert_allclose(dr[1,2],-.5*gamma*rho[1,2])
    np.testing.assert_allclose(np.trace(dr),0,atol=1e-14)


def test_trap_and_dephasing_are_different_spectral_mechanisms():
    grid=np.linspace(.94,1.24,4)
    cfg=DimerSpectrumConfig(grid,grid,eta=.04)
    q=[.025,.014,.022,.012]
    trap=np.asarray(build_structural_observable(cfg,"C")(q,.9))
    dephase=np.asarray(build_structural_observable(cfg,"B")(q,.9))
    assert np.linalg.norm(trap-dephase)>1e-5
