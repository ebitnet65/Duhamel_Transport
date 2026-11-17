"""Two-control full 2D rephasing response and state-generator geometry."""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.linalg
from phase_matched_response import liouvillian, steady_state, vec, mat, G_freq, get_sector_pathways, omega1_sign_for_sector, interaction_super, mu

W0=1.0
GPHI_REF=0.1
GAMMA1=0.02
ETA=1e-3
THETAS=np.array([0.0,np.pi/4,np.pi/2])
GPHIS=np.array([0.05,0.1,0.2])
T2S=np.array([0.2,1.0,5.0])

def stationary(L):
    # Replace one dependent trace-preserving equation by Tr(rho)=1.
    M=L.copy()
    M[-1,:]=np.array([1,0,0,1],complex)
    b=np.zeros(4,complex); b[-1]=1
    return np.linalg.solve(M,b)

def sector_response_map(theta,gamma_phi,t2,omegas,sector):
    L=liouvillian(w0=W0,gamma_phi=gamma_phi,theta=theta,gamma1=GAMMA1)
    # Fixed aligned-reference preparation. At theta=0 the stationary state is
    # |g><g| for every positive gamma_phi in this scan.
    Lref=liouvillian(w0=W0,gamma_phi=gamma_phi,theta=0.,gamma1=GAMMA1)
    rho0=stationary(Lref)
    U=scipy.linalg.expm(L*t2)
    # The response module uses G(w)=-[L+i*w-eta]^-1. Positive displayed
    # excitation/detection frequencies map to signed Fourier arguments -w;
    # the sector-specific omega1 sign then distinguishes RP from NRP.
    signed_omegas=-np.asarray(omegas)
    s1=omega1_sign_for_sector(sector)
    G1=np.asarray([G_freq(L,s1*w,eta=ETA) for w in signed_omegas])
    G3=np.asarray([G_freq(L,w,eta=ETA) for w in signed_omegas])
    D=np.array([np.trace(mu@mat(np.eye(4,dtype=complex)[:,k])) for k in range(4)])
    S=np.zeros((len(omegas),len(omegas)),complex)
    for path in get_sector_pathways(sector):
        J1,J2,J3=[interaction_super(*x) for x in path]
        first=np.einsum("wij,j->wi",G1,J1@rho0)
        second=np.einsum("ij,wj->wi",J2,first)
        waiting=np.einsum("ij,wj->wi",U,second)
        third=np.einsum("ij,wj->wi",J3,waiting)
        for i,state in enumerate(third):
            detected=np.einsum("wij,j->wi",G3,state)
            S[i,:]+=D@detected.T
    return S,L,stationary(L)

def embedding(q):
    theta, gscaled=q
    gamma_phi=GPHI_REF*gscaled
    L=liouvillian(w0=W0,gamma_phi=gamma_phi,theta=theta,gamma1=GAMMA1)
    rho=stationary(L)
    return np.concatenate((rho.real,rho.imag,L.real.ravel(),L.imag.ravel()))

def derivs_embedding(q,h):
    p=np.array(q,float); h1,h2=h
    e1=np.array([h1,0.]); e2=np.array([0.,h2])
    x=embedding(p)
    j1=(embedding(p+e1)-embedding(p-e1))/(2*h1)
    j2=(embedding(p+e2)-embedding(p-e2))/(2*h2)
    h11=(embedding(p+e1)-2*x+embedding(p-e1))/h1**2
    h22=(embedding(p+e2)-2*x+embedding(p-e2))/h2**2
    h12=(embedding(p+e1+e2)-embedding(p+e1-e2)-embedding(p-e1+e2)+embedding(p-e1-e2))/(4*h1*h2)
    return np.column_stack((j1,j2)), np.stack((h11,h12,h22))

def geometry_point(theta,gamma_phi,h=(1e-4,1e-4)):
    q=np.array([theta,gamma_phi/GPHI_REF])
    J,H=derivs_embedding(q,h)
    G=J.T@J
    ev=np.linalg.eigvalsh(G)
    inv=np.linalg.inv(G)
    # Levi-Civita connection from the embedded surface's tangential acceleration.
    Gamma=np.empty((2,2,2))
    for mu in range(2):
      for nu in range(2):
        Hmunu=H[0] if (mu,nu)==(0,0) else H[2] if (mu,nu)==(1,1) else H[1]
        Gamma[:,mu,nu]=inv@(J.T@Hmunu)
    B=np.empty((2,2,J.shape[0]))
    for a in range(2):
      for b in range(2):
        Hmunu=H[0] if (a,b)==(0,0) else H[2] if (a,b)==(1,1) else H[1]
        B[a,b]=Hmunu-J@Gamma[:,a,b]
    K=float((B[0,0]@B[1,1]-B[0,1]@B[0,1])/np.linalg.det(G))
    L=liouvillian(w0=W0,gamma_phi=gamma_phi,theta=theta,gamma1=GAMMA1)
    rho=mat(stationary(L))
    eigL=np.linalg.eigvals(L)
    decay=np.sort(-np.real(eigL[np.real(eigL)<-1e-9]))
    return dict(theta=float(theta),gamma_phi=float(gamma_phi),metric=G.tolist(),
      metric_eigenvalues=ev.tolist(),
      christoffel=Gamma.tolist(),gaussian_curvature=K,scalar_curvature=2*K,
      min_density_eigenvalue=float(np.min(np.linalg.eigvalsh(rho)).real),
      trace=float(np.trace(rho).real),steady_residual=float(np.linalg.norm(L@vec(rho))),
      spectral_gap=float(np.min(decay)) if decay.size else 0.0,
      condition_number=float(ev[-1]/ev[0]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-omega",type=int,default=61)
    ap.add_argument("--out",default="two_control_spectroscopy")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    omegas=np.linspace(.25,1.75,args.n_omega)
    rp_maps=np.empty((len(GPHIS),len(THETAS),len(T2S),len(omegas),len(omegas)),complex)
    nrp_maps=np.empty_like(rp_maps)
    absorptive_maps=np.empty(rp_maps.shape,float)
    records=[]; t0=time.time()
    for ig,gphi in enumerate(GPHIS):
      for it,theta in enumerate(THETAS):
       for jt,t2 in enumerate(T2S):
        RP,L,rho_vec=sector_response_map(theta,gphi,t2,omegas,"RP")
        NRP,_,_=sector_response_map(theta,gphi,t2,omegas,"NRP")
        rho=mat(rho_vec)
        rp_maps[ig,it,jt]=RP
        nrp_maps[ig,it,jt]=NRP
        Sabs=-(RP+NRP).real
        absorptive_maps[ig,it,jt]=Sabs
        ref=absorptive_maps[ig,0,jt]
        norm=np.trapezoid(np.trapezoid(abs(ref)**2,omegas,axis=1),omegas)
        err=np.trapezoid(np.trapezoid(abs(Sabs-ref)**2,omegas,axis=1),omegas)
        # population/coherence coupling norm in vec order [gg,eg,ge,ee].
        pop=np.array([0,3]); coh=np.array([1,2])
        mix=float(np.linalg.norm(L[np.ix_(pop,coh)])+np.linalg.norm(L[np.ix_(coh,pop)]))
        evalL=np.linalg.eigvals(L); decay=np.sort(-np.real(evalL[np.real(evalL)<-1e-9]))
        records.append(dict(theta=float(theta),gamma_phi=float(gphi),t2=float(t2),
          absorptive_l2=float(np.sqrt(np.trapezoid(np.trapezoid(abs(Sabs)**2,omegas,axis=1),omegas))),
          relative_map_change_from_aligned=float(np.sqrt(err/max(norm,1e-30))),
          population_coherence_block_norm=mix,
          stationary_min_eigenvalue=float(np.min(np.linalg.eigvalsh(rho)).real),
          stationary_trace=float(np.trace(rho).real),
          steady_residual=float(np.linalg.norm(L@vec(rho))),
          spectral_gap=float(np.min(decay)) if decay.size else 0.0))
    # Compare dephasing-rate changes to the gamma_phi=0.10 map at fixed theta,t2.
    for record in records:
        it=int(np.flatnonzero(THETAS==record["theta"])[0])
        jt=int(np.flatnonzero(T2S==record["t2"])[0])
        ig=int(np.flatnonzero(GPHIS==record["gamma_phi"])[0])
        rate_ref=absorptive_maps[1,it,jt]
        den=np.trapezoid(np.trapezoid(abs(rate_ref)**2,omegas,axis=1),omegas)
        err=np.trapezoid(np.trapezoid(abs(absorptive_maps[ig,it,jt]-rate_ref)**2,omegas,axis=1),omegas)
        record["relative_map_change_from_reference_rate"]=float(np.sqrt(err/max(den,1e-30)))
    np.savez_compressed(out/"response_maps.npz",omegas=omegas,thetas=THETAS,gamma_phis=GPHIS,t2s=T2S,
      gamma1=GAMMA1,gamma_phi_reference=GPHI_REF,w0=W0,eta=ETA,
      rephasing=rp_maps,nonrephasing=nrp_maps,absorptive=absorptive_maps)
    # Geometry grid with finite-difference step convergence at one interior point.
    geom=[]
    for theta in np.linspace(.05,np.pi/2-.05,13):
      for gphi in np.linspace(.05,.2,9):
        geom.append(geometry_point(theta,gphi))
    conv=[]
    for h in [1e-3,3e-4,1e-4,3e-5,1e-5]:
      g=geometry_point(np.pi/4,.1,h=(h,h))
      conv.append(dict(step=h,metric_eigenvalues=g["metric_eigenvalues"],
                       gaussian_curvature=g["gaussian_curvature"],christoffel=g["christoffel"]))
    summary=dict(conventions=dict(model="rotated dephasing TLS",w0=W0,gamma1=GAMMA1,
      gamma_phi_reference=GPHI_REF,controls=["theta","gamma_phi/gamma_phi_reference"],
      initial="fixed aligned stationary state",frequency_range=[.25,1.75],eta=ETA,
      waiting_times=T2S.tolist(),
      spectral_convention="positive display axes use signed Fourier arguments -omega1,-omega3; absorptive=-Re(RP+NRP)",
      generator_embedding="real and imaginary entries of full Liouvillian",
      generator_scale=1.0),spectra=records,geometry=geom,geometry_step_convergence=conv,
      elapsed_s=time.time()-t0)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    # Summary figures use shared color limits within each plotted row.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ig=1; jt=1
    extent=[omegas[0],omegas[-1],omegas[0],omegas[-1]]
    full_maps=absorptive_maps[ig,:,jt]
    full_min=float(full_maps.real.min()); full_max=float(full_maps.real.max())
    diffmax=float(max(np.max(abs(full_maps[i]-full_maps[0])) for i in range(len(THETAS))))
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i,theta in enumerate(THETAS):
      im0=axes[0,i].imshow(full_maps[i].real,origin="lower",extent=extent,aspect="auto",cmap="RdBu_r",vmin=full_min,vmax=full_max)
      axes[0,i].set_title(rf"$\theta={theta:.3f}$")
      axes[0,i].set_xlabel(r"$\omega_1$")
      axes[0,i].set_ylabel(r"$\omega_3$")
      delta=full_maps[i]-full_maps[0]
      im1=axes[1,i].imshow(abs(delta),origin="lower",extent=extent,aspect="auto",cmap="magma",vmin=0,vmax=diffmax)
      axes[1,i].set_xlabel(r"$\omega_1$")
      axes[1,i].set_ylabel(r"$\omega_3$")
    fig.colorbar(im0,ax=axes[0,:].tolist(),shrink=.82,label=r"$S_{\rm abs}^{(3)}$")
    fig.colorbar(im1,ax=axes[1,:].tolist(),shrink=.82,label=r"$|\Delta S_{\rm abs}^{(3)}|$")
    fig.suptitle(r"Absorptive 2D spectra: $\gamma_\phi=0.1$, $t_2=1$")
    fig.savefig(out/"figure_full_2d_maps.png",dpi=180)
    # Waiting-time sensitivity at maximal misalignment and reference dephasing.
    deltas=np.asarray([absorptive_maps[1,2,j]-absorptive_maps[1,0,j] for j in range(len(T2S))])
    real_min=float(deltas.real.min()); real_max=float(deltas.real.max())
    delta_max=float(np.max(abs(deltas)))
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,t2 in enumerate(T2S):
      im0=axes[0,j].imshow(deltas[j].real,origin="lower",extent=extent,aspect="auto",cmap="RdBu_r",vmin=real_min,vmax=real_max)
      axes[0,j].set_title(rf"$t_2={t2:g}$")
      axes[0,j].set_xlabel(r"$\omega_1$"); axes[0,j].set_ylabel(r"$\omega_3$")
      im1=axes[1,j].imshow(abs(deltas[j]),origin="lower",extent=extent,aspect="auto",cmap="magma",vmin=0,vmax=delta_max)
      axes[1,j].set_xlabel(r"$\omega_1$"); axes[1,j].set_ylabel(r"$\omega_3$")
    fig.colorbar(im0,ax=axes[0,:].tolist(),shrink=.82,label=r"$\Delta S_{\rm abs}^{(3)}$")
    fig.colorbar(im1,ax=axes[1,:].tolist(),shrink=.82,label=r"$|\Delta S_{\rm abs}^{(3)}|$")
    fig.suptitle(r"Waiting-time dependence: $\theta=\pi/2$, $\gamma_\phi=0.1$")
    fig.savefig(out/"figure_waiting_time_maps.png",dpi=180)
    print(f"{len(records)} maps; grid {args.n_omega}x{args.n_omega}; geometry points {len(geom)}; elapsed {time.time()-t0:.2f}s")
    for g in conv: print("geometry convergence",g)
    print("representative spectral rows:")
    for r in records:
      if r["gamma_phi"]==.1 and r["t2"]==1.0:
        print(r)
if __name__=="__main__": main()
