#!/usr/bin/env python3
"""Reproducible sensitivity and hidden-trap benchmark for the dimer model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from dimer_spectral_observable import DimerSpectrumConfig, build_dimer_spectral_observable, pack_complex_spectrum
from dimer_structural_discrimination import (
    build_structural_observable, finite_difference_jacobian, whitened_diagnostics,
)
from spectral_observability import generalized_observability_modes

jax.config.update("jax_enable_x64", True)
Q0=np.array([.025,.014,.022])
PARAMETER_NAMES=("J12", "k12", "gamma_phi")
PARAMETER_SCALES=np.array([.05,.015,.025])


def _packed(z):
    z=np.asarray(z)
    return np.r_[z.real.ravel(),z.imag.ravel()]


def _unpacked(x, shape):
    x=np.asarray(x,float); n=int(np.prod(shape))
    return (x[:n]+1j*x[n:]).reshape(shape)


def _fit_model(observable, data, times, model, sigma, max_nfev):
    nparam=3 if model=="A" else 4
    lo=np.array([-.1,1e-5,1e-5]+([] if model=="A" else [0.]))
    hi=np.array([.1,.03,.04]+([] if model=="A" else [.08]))
    start=np.r_[Q0] if model=="A" else np.r_[Q0, .005 if model=="B" else .01]
    obs=observable[model]
    funs=[]
    for t in times:
        def f(x,t=t):
            value=obs(jnp.concatenate((x,jnp.zeros((1,),dtype=x.dtype))) if model=="A" else x,t)
            flat=jnp.ravel(value)
            return jnp.concatenate((jnp.real(flat),jnp.imag(flat)))
        funs.append(jax.jit(f))
    jacs=[jax.jit(jax.jacfwd(f)) for f in funs]
    target=np.concatenate([_packed(data[float(t)]) for t in times])
    def residual(x):
        return (np.concatenate([np.asarray(f(jnp.asarray(x))) for f in funs])-target)/sigma
    def jac(x):
        return np.vstack([np.asarray(j(jnp.asarray(x))) for j in jacs])/sigma
    res=least_squares(residual,start,jac=jac,bounds=(lo,hi),max_nfev=max_nfev,
                      xtol=2e-10,ftol=2e-10,gtol=2e-10,x_scale="jac")
    sv=np.linalg.svd(jac(res.x),compute_uv=False)
    rank=int(np.linalg.matrix_rank(jac(res.x)))
    dof=max(1,target.size-nparam)
    cov=np.linalg.pinv(jac(res.x).T@jac(res.x),rcond=1e-11)
    redchi=float(np.dot(res.fun,res.fun)/dof)
    # Local Gaussian interval; boundary-aware one-sided estimates are flagged.
    se=np.sqrt(np.maximum(np.diag(cov)*max(redchi,1.),0.))
    return {"x":res.x,"success":bool(res.success),"message":res.message,
            "cost":float(np.dot(res.fun,res.fun)),"nfev":int(res.nfev),
            "rank":rank,"singular_values":sv,"condition":float(sv[0]/sv[rank-1]) if rank else float("inf"),
            "stderr":se,"ci95":np.c_[res.x-1.96*se,res.x+1.96*se],
            "residual":res.fun,"reduced_chi2":redchi,"dof":dof}


def run(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(args.omega_min,args.omega_max,args.nfreq)
    cfg=DimerSpectrumConfig(grid,grid,eta=args.eta)
    base=build_dimer_spectral_observable(cfg)
    rp=build_dimer_spectral_observable(replace(cfg,sectors=("RP",)))
    nrp=build_dimer_spectral_observable(replace(cfg,sectors=("NRP",)))
    q=jnp.asarray(Q0,dtype=jnp.float64)
    times=np.asarray(args.waiting_times,float)
    base_specs={float(t):np.asarray(base(q,t)) for t in times}
    rp_specs={float(t):np.asarray(rp(q,t)) for t in times}
    nrp_specs={float(t):np.asarray(nrp(q,t)) for t in times}
    # Baseline parametric sensitivity atlas and coordinate-normalized modes.
    base_jac={float(t):np.asarray(jax.jacfwd(lambda x:base(x,t))(q)) for t in times}
    diagnostics={float(t):whitened_diagnostics(base_jac[float(t)],args.sigma) for t in times}
    physical=np.diag(1/PARAMETER_SCALES**2)
    gen_modes={float(t):generalized_observability_modes(d["information"],physical)
               for t,d in diagnostics.items()}
    fd=finite_difference_jacobian(base,Q0,float(times[len(times)//2]))
    ad=base_jac[float(times[len(times)//2])]
    fd_rel=float(np.linalg.norm(ad-fd)/max(np.linalg.norm(ad),1e-30))

    models={m:build_structural_observable(cfg,m) for m in "ABC"}
    traps=np.asarray(args.trap_rates,float)
    truth_by_kd={}; single_fits={}; residuals={}
    # All candidate spectra share the same baseline microscopic truth.
    for kd in traps:
        truth_by_kd[float(kd)]={float(t):np.asarray(models["C"](jnp.r_[q,kd],t)) for t in times}
        if kd>0:
            for t in times:
                key=f"kd={kd:g},t={t:g}"
                fit=_fit_model(models, {float(t):truth_by_kd[float(kd)][float(t)]},
                               [float(t)],"A",args.sigma,args.max_nfev)
                single_fits[key]=fit
                predicted=np.asarray(models["A"](jnp.r_[fit["x"],0.],t))
                residuals[key]=predicted-truth_by_kd[float(kd)][float(t)]

    # Small-trap bias and stable noise-whitened tangent/orthogonal split.
    tangent_results={}
    jdc=jax.jacfwd(lambda x:models["C"](x,float(times[0])))
    for t in times:
        jl=base_jac[float(t)].reshape((-1,3))
        jl=np.vstack((jl.real,jl.imag))/args.sigma
        jc=np.asarray(jax.jacfwd(lambda x:models["C"](x,float(t)))(jnp.r_[q,0.]))
        jd=jc[...,3].reshape(-1)
        jd_w=np.r_[jd.real,jd.imag]/args.sigma
        u,s,_=np.linalg.svd(jl,full_matrices=False)
        tol=max(jl.shape)*np.finfo(float).eps*s[0]
        rank=int(np.sum(s>tol)); qbasis=u[:,:rank]
        projector=qbasis@qbasis.T
        ider=jl.T@jl
        bias_per_rate=np.linalg.pinv(ider,rcond=1e-12)@(jl.T@jd_w)
        positive_traps=traps[traps>0]
        kd_probe=float(positive_traps[len(positive_traps)//2]) if positive_traps.size else .01
        kd_small=float(positive_traps[0]) if positive_traps.size else .001
        perturb=_packed(truth_by_kd[kd_probe][float(t)])-_packed(base_specs[float(t)])
        perturb_w=perturb/args.sigma
        absorb=projector@perturb_w
        orth=perturb_w-absorb
        tangent_results[str(float(t))]={"Jd_norm":float(np.linalg.norm(jd_w)),
            "tangent_rank":rank,"bias_per_unit_kd":bias_per_rate.tolist(),
            "small_probe_kd":kd_small,"first_order_bias_at_small_probe":(kd_small*bias_per_rate).tolist(),
            "moderate_probe_kd":kd_probe,"perturbation_norm":float(np.linalg.norm(perturb_w)),
            "tangent_absorbable_norm":float(np.linalg.norm(absorb)),
            "orthogonal_residual_norm":float(np.linalg.norm(orth)),
            "orthogonal_fraction":float(np.linalg.norm(orth)/max(np.linalg.norm(perturb_w),1e-30)),
            "orthogonal_residual_spectrum":{"real":_unpacked(orth*args.sigma,base_specs[float(t)].shape).real.tolist(),"imag":_unpacked(orth*args.sigma,base_specs[float(t)].shape).imag.tolist()}}

    # Conditional information for detecting a nonnegative trap rate at kd=0.
    trap_jac=np.asarray(jax.jacfwd(lambda x:models["C"](x,float(times[len(times)//2])))(jnp.r_[q,0.]))
    jt=trap_jac.reshape((-1,4)); jt=np.vstack((jt.real,jt.imag))/args.sigma
    it=jt.T@jt; ill=it[:3,:3]; ild=it[:3,3]; idl=it[3,:3]; idd=it[3,3]
    conditional=float(idd-idl@np.linalg.pinv(ill,rcond=1e-12)@ild)

    # Shared-parameter multi-time comparison at moderate trap strength.
    kd_ref=float(traps[min(len(traps)-1,max(0,len(traps)//2))])
    if kd_ref==0 and len(traps)>1: kd_ref=float(traps[-1])
    truth=truth_by_kd[kd_ref]
    model_fits={}
    for m in "ABC":
        fit=_fit_model(models,truth,times,m,args.sigma,args.max_nfev)
        model_fits[m]=fit
        fit["AIC"] = fit["cost"]+2*(3 if m=="A" else 4)
        fit["heldout"]={}
        for t in np.linspace(times[0],times[-1],max(3,len(times)+1)):
            y=np.asarray(models["C"](jnp.r_[q,kd_ref],float(t)))
            x=fit["x"] if m!="A" else np.r_[fit["x"],0.]
            yp=np.asarray(models[m](x,float(t)))
            fit["heldout"][str(float(t))]=float(np.linalg.norm(yp-y)/max(np.linalg.norm(y),1e-30))
        fit["residual_maps"]={str(float(t)):{"real":(np.asarray(models[m](fit["x"] if m!="A" else jnp.r_[fit["x"],0.],float(t)))-truth[float(t)]).real.tolist(),"imag":(np.asarray(models[m](fit["x"] if m!="A" else jnp.r_[fit["x"],0.],float(t)))-truth[float(t)]).imag.tolist()} for t in times}

    # Reproducible complex Gaussian noise recovery, single-slice vs joint-time.
    rng_seeds=[11,29,47]
    noise_recovery={}
    noise_truth=truth_by_kd[0.0]
    for seed in rng_seeds:
        rng=np.random.default_rng(seed)
        noisy={float(t):noise_truth[float(t)]+args.sigma*(rng.standard_normal(noise_truth[float(t)].shape)+1j*rng.standard_normal(noise_truth[float(t)].shape)) for t in times}
        one_t=float(times[len(times)//2])
        one=_fit_model(models,noisy,[one_t],"A",args.sigma,args.max_nfev)
        joint_noise=_fit_model(models,noisy,times,"A",args.sigma,args.max_nfev)
        noise_recovery[str(seed)]={"single_time":float(one_t),"single_endpoint":one["x"].tolist(),"single_stderr":one["stderr"].tolist(),"single_rank":one["rank"],"single_condition":one["condition"],"single_cost":one["cost"],"joint_endpoint":joint_noise["x"].tolist(),"joint_stderr":joint_noise["stderr"].tolist(),"joint_rank":joint_noise["rank"],"joint_condition":joint_noise["condition"],"joint_cost":joint_noise["cost"]}
    reverse_fit=_fit_model(models,noise_truth,[float(t) for t in times],"C",args.sigma,args.max_nfev)

    # Joint-information design score; compare equal numbers of waiting times.
    candidates=[]
    for i in range(len(times)):
        for j in range(i+1,len(times)):
            ii=diagnostics[float(times[i])]["information"]+diagnostics[float(times[j])]["information"]
            eig=np.linalg.eigvalsh(ii)
            candidates.append({"times":[float(times[i]),float(times[j])],
                               "eigenvalues":eig.tolist(),"minimum_eigenvalue":float(eig[0]),
                               "logdet":float(np.linalg.slogdet(ii+1e-16*np.eye(3))[1]),
                               "condition":float(eig[-1]/eig[0]) if eig[0]>0 else float("inf")})
    candidates.sort(key=lambda x:(x["minimum_eigenvalue"],x["logdet"]),reverse=True)

    # Local trap-vs-dephasing design score after allowing common dimer
    # parameters to refit. Orthogonal hidden-channel directions are compared
    # for equal numbers of measured waiting-time slices.
    structural_jac={}
    for m in "BC":
        structural_jac[m]={}
        for t in times:
            structural_jac[m][float(t)]=np.asarray(jax.jacfwd(lambda x:models[m](x,float(t)))(jnp.r_[q,0.]))
    mechanism_design=[]
    for count in (1,2):
        import itertools
        for selected in itertools.combinations([float(t) for t in times],count):
            # Each block is real/imaginary stacked and noise whitened.
            jblocks=[]; cblocks=[]; bblocks=[]
            for t in selected:
                a=base_jac[t].reshape((-1,3)); jblocks.append(np.vstack((a.real,a.imag))/args.sigma)
                for m,store in (("C",cblocks),("B",bblocks)):
                    d=structural_jac[m][t][...,3].reshape(-1); store.append(np.r_[d.real,d.imag]/args.sigma)
            jmat=np.vstack(jblocks); u,s,_=np.linalg.svd(jmat,full_matrices=False)
            rr=int(np.sum(s>max(jmat.shape)*np.finfo(float).eps*s[0])); basis=u[:,:rr]
            vc=np.concatenate(cblocks); vb=np.concatenate(bblocks)
            pc=vc-basis@(basis.T@vc); pb=vb-basis@(basis.T@vb)
            nc=float(np.linalg.norm(pc)); nb=float(np.linalg.norm(pb))
            cosine=float(np.dot(pc,pb)/max(nc*nb,1e-30))
            cosine=float(np.clip(cosine,-1.,1.))
            contrast=float(min(nc,nb)*np.sqrt(max(0.,1-cosine*cosine)))
            mechanism_design.append({"times":list(selected),"trap_orthogonal_norm":nc,
                "dephasing_orthogonal_norm":nb,"cosine":cosine,
                "trap_dephasing_contrast_score":contrast})
    mechanism_design.sort(key=lambda x:x["trap_dephasing_contrast_score"],reverse=True)

    arrays={"omega_grid":grid,"waiting_times":times,"q0":Q0,
            "spectra":np.array([base_specs[float(t)] for t in times]),
            "spectra_RP":np.array([rp_specs[float(t)] for t in times]),
            "spectra_NRP":np.array([nrp_specs[float(t)] for t in times]),
            "jacobians":np.array([base_jac[float(t)] for t in times]),
            "trap_rates":traps,
            "trap_truth_spectra":np.array([[truth_by_kd[float(kd)][float(t)] for t in times] for kd in traps]),
            "parameter_scales":PARAMETER_SCALES}
    np.savez_compressed(out/"spectral_sensitivity_benchmark.npz",**arrays)

    def fit_serial(f):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items() if k not in {"residual"}}
    report={"baseline_parameters":dict(zip(PARAMETER_NAMES,Q0)),"parameter_units":{"J12":"energy","k12":"inverse time","gamma_phi":"inverse time"},
      "grid":{"frequency":grid.tolist(),"waiting_times":times.tolist(),"eta":args.eta,"sectors":["RP","NRP","RP+NRP"],"complex_real_imag_stacked":True},
      "noise":{"independent_complex_gaussian_cartesian_sigma":args.sigma,"interpretation":"same sigma for each real and imaginary sample"},
      "metric":{"definition":"G=diag(1/s_i^2), with s=(0.05 energy, 0.015 rate, 0.025 rate)","normalization":"dimensionless unit displacement in each stated reference scale"},
      "autodiff_centered_difference_relative_error":fd_rel,
      "baseline_by_waiting_time":{str(t):{"singular_values":diagnostics[float(t)]["singular_values"].tolist(),"rank":diagnostics[float(t)]["rank"],"condition":diagnostics[float(t)]["condition"],"information":diagnostics[float(t)]["information"].tolist(),"parameter_correlation":diagnostics[float(t)]["correlation"].tolist(),"generalized_eigenvalues":gen_modes[float(t)].eigenvalues.tolist(),"generalized_eigenvectors":gen_modes[float(t)].eigenvectors.tolist()} for t in times},
      "two_time_design_candidates":candidates,"single_time_baseline_fits":{k:fit_serial(v) for k,v in single_fits.items()},
      "trap_vs_dephasing_design_candidates":mechanism_design,
      "hidden_channel_tangent_decomposition":tangent_results,
      "conditional_trap_information_at_zero":{"waiting_time":float(times[len(times)//2]),"I_d_given_lambda":conditional,"boundary":"k_d>=0; symmetric Gaussian intervals at zero are invalid. Use a one-sided profile/bootstrap or boundary-aware likelihood ratio for a formal claim."},
      "baseline_noise_recovery":{"seeds":rng_seeds,"results":noise_recovery},
      "reverse_overparameterization":{"truth":"A (kd=0)","fit_model":"C","fit":fit_serial(reverse_fit),"boundary_interpretation":"A symmetric interval is inappropriate at kd=0; the point estimate and conditional information are reported, while formal evidence requires a one-sided profile or boundary-aware likelihood ratio."},
      "joint_model_comparison":{"truth_kd":kd_ref,"fits":{m:fit_serial(f) for m,f in model_fits.items()},"interpretation":"AIC is reported as a noise-scaled residual plus 2p; it is a local Gaussian diagnostic, not proof of microscopic mechanism."},
      "limitations":["The optical response uses the exact active-block projection of a terminal dark sink; the sink population is validated with the full trace-preserving generator.","Confidence intervals use local linear covariance and are unreliable at bounds or for strongly nonlinear/weakly identified directions.","AIC assumes independent Gaussian Cartesian real/imaginary noise with the configured common sigma."]}
    (out/"spectral_sensitivity_benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=True)+"\n")

    # Publication-style overview figure: baseline, coordinate derivatives,
    # and hidden-channel reduced-model drift across waiting time.
    fig,axs=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    mid=float(times[len(times)//2]); spec=base_specs[mid]
    axs[0,0].imshow(spec.real,origin="lower",aspect="auto",extent=[grid[0],grid[-1],grid[0],grid[-1]])
    axs[0,0].set_title(f"Baseline Re(S), t₂={mid:g}")
    derivative_units=("signal / energy","signal / rate","signal / rate")
    for k,name in enumerate(PARAMETER_NAMES):
        derivative=base_jac[mid][...,k].real
        limit=max(float(np.max(np.abs(derivative))),np.finfo(float).tiny)
        im=axs[0,k+1].imshow(derivative,origin="lower",aspect="auto",extent=[grid[0],grid[-1],grid[0],grid[-1]],cmap="coolwarm",vmin=-limit,vmax=limit)
        axs[0,k+1].set_title(f"∂ Re(S) / ∂ {name}\nrange ±{limit:.2g} {derivative_units[k]}")
        fig.colorbar(im,ax=axs[0,k+1],shrink=.78)
    for ax in axs[0]: ax.set_xlabel("ω₁"); ax.set_ylabel("ω₃")
    kd_plot=float(traps[min(len(traps)-1,max(1,len(traps)//2))])
    for j,name in enumerate(PARAMETER_NAMES):
        estimates=[]; lows=[]; highs=[]
        for t in times:
            fit=single_fits[f"kd={kd_plot:g},t={t:g}"]
            estimates.append(fit["x"][j]); lows.append(fit["ci95"][j,0]); highs.append(fit["ci95"][j,1])
        axs[1,0].errorbar(times,estimates,yerr=[np.asarray(estimates)-lows,np.asarray(highs)-estimates],marker="o",capsize=3,label=name)
        axs[1,0].axhline(Q0[j],color=f"C{j}",ls=":",alpha=.55)
    axs[1,0].set_title(f"A fits vs t₂, truth kd={kd_plot:g} (95% local CI)")
    axs[1,0].set_xlabel("t₂"); axs[1,0].legend(fontsize=8)
    names=list("ABC"); costs=[model_fits[m]["cost"] for m in names]
    axs[1,1].bar(names,costs); axs[1,1].set_yscale("log"); axs[1,1].set_ylim(bottom=1e-12)
    axs[1,1].set_title(f"Joint fit weighted SSE (truth kd={kd_ref:g})")
    axs[1,1].set_ylabel("χ²")
    singular=np.asarray([diagnostics[float(t)]["singular_values"] for t in times])
    for i in range(singular.shape[1]): axs[1,2].plot(times,singular[:,i],"o-",label=f"s{i+1}")
    axs[1,2].set_yscale("log"); axs[1,2].set_title("Baseline Jacobian singular values")
    axs[1,2].set_xlabel("t₂"); axs[1,2].set_ylabel("singular value"); axs[1,2].legend(fontsize=8)
    mode=gen_modes[mid].eigenvectors[:,-1]
    perturb=np.tensordot(base_jac[mid],mode,axes=([-1],[0]))
    axs[1,3].imshow(perturb.real,origin="lower",aspect="auto",extent=[grid[0],grid[-1],grid[0],grid[-1]],cmap="coolwarm")
    axs[1,3].set_title("Least observable generalized mode")
    axs[1,3].set_xlabel("ω₁"); axs[1,3].set_ylabel("ω₃")
    fig.savefig(out/"spectral_sensitivity_model_discrimination.png",dpi=180)
    plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="docs/benchmarks/spectral_sensitivity_model_discrimination")
    parser.add_argument("--nfreq",type=int,default=5)
    parser.add_argument("--omega-min",type=float,default=.90)
    parser.add_argument("--omega-max",type=float,default=1.28)
    parser.add_argument("--waiting-times",type=float,nargs="+",default=[.2,.8,2.5,6.])
    parser.add_argument("--trap-rates",type=float,nargs="+",default=[0.,.0003,.003,.012,.04])
    parser.add_argument("--eta",type=float,default=.035)
    parser.add_argument("--sigma",type=float,default=1e-3)
    parser.add_argument("--max-nfev",type=int,default=60)
    args=parser.parse_args()
    report=run(args)
    print(json.dumps({"output_dir":args.output_dir,"fd_relative_error":report["autodiff_centered_difference_relative_error"],"joint_fit_costs":{k:v["cost"] for k,v in report["joint_model_comparison"]["fits"].items()}},indent=2))


if __name__=="__main__": main()
