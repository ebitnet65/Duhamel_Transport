#!/usr/bin/env python3
"""Adversarial local stress test for trap/dephasing discrimination.

The test compares equal-pixel-count spectral windows, single and joint waits,
and several noise levels.  It uses whitened tangent-space projections so each
candidate may refit the common dimer parameters and its own extra rate.  A
Monte Carlo score checks how often the pure-dephasing candidate beats the
baseline under a first-order dark-trap truth.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dimer_spectral_observable import DimerSpectrumConfig
from dimer_structural_discrimination import build_structural_observable

jax.config.update("jax_enable_x64", True)
Q0=np.array([.025,.014,.022])


def _real_jac(jac, mask, absorptive=False):
    selected=np.asarray(jac).reshape((-1,jac.shape[-1]))[mask]
    return selected.real if absorptive else np.vstack((selected.real,selected.imag))


def _real_signal(signal, mask, absorptive=False):
    selected=np.asarray(signal).ravel()[mask]
    return selected.real if absorptive else np.r_[selected.real,selected.imag]


def _basis(matrix):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    tol=max(matrix.shape)*np.finfo(float).eps*(s[0] if s.size else 0.)
    rank=int(np.sum(s>tol))
    return u[:,:rank],s,rank


def _windows(n):
    if n<5:
        raise ValueError("stress grid requires at least five frequencies")
    windows={"center_3x3":np.zeros((n,n),bool),
             "low_3x3":np.zeros((n,n),bool),
             "high_3x3":np.zeros((n,n),bool),
             "crosspeak_9":np.zeros((n,n),bool),
             "full":np.ones((n,n),bool)}
    windows["center_3x3"][n//2-1:n//2+2,n//2-1:n//2+2]=True
    windows["low_3x3"][:3,:3]=True
    windows["high_3x3"][-3:,-3:]=True
    coords=[(i,j) for i in range(n) for j in range(n)]
    # Nine strongest off-diagonal pixels, chosen once from geometry only.
    for i,j in sorted(coords,key=lambda ij:abs(ij[0]-ij[1]),reverse=True)[:9]:
        windows["crosspeak_9"][i,j]=True
    return windows


def run(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(args.omega_min,args.omega_max,args.nfreq)
    config=DimerSpectrumConfig(grid,grid,eta=args.eta)
    models={m:build_structural_observable(config,m) for m in "ABC"}
    q=jnp.asarray(Q0,dtype=jnp.float64)
    times=np.asarray(args.waiting_times,float)
    spectra={float(t):np.asarray(models["A"](jnp.r_[q,0.],float(t))) for t in times}
    jac={m:{} for m in "ABC"}
    for m in "ABC":
        for t in times:
            jac[m][float(t)]=np.asarray(jax.jacfwd(lambda x:models[m](x,float(t)))(jnp.r_[q,0.]))
    windows=_windows(args.nfreq)
    # Experiment sets are compared at identical numbers of waiting-time slices.
    time_sets={"early":(float(times[0]),),"late":(float(times[-1]),),
               "early_late":(float(times[0]),float(times[-1])),
               "all":tuple(map(float,times))}
    trap_rates=[0.,args.weak_trap,args.moderate_trap]
    noise_levels=list(args.relative_noise)
    seeds=list(range(args.seed,args.seed+args.nmc))
    records=[]
    for wname,wmask in windows.items():
        pixelmask=wmask.ravel()
        npix=int(pixelmask.sum())
        for tname,selected_times in time_sets.items():
            a_blocks=[]; b_blocks=[]; c_blocks=[]; signal_blocks=[]
            for t in selected_times:
                a_blocks.append(_real_jac(jac["A"][t][...,:3],pixelmask,args.absorptive))
                b_blocks.append(_real_jac(jac["B"][t][...,3:4],pixelmask,args.absorptive))
                c_blocks.append(_real_jac(jac["C"][t][...,3:4],pixelmask,args.absorptive))
                signal_blocks.append(_real_signal(spectra[t],pixelmask,args.absorptive))
            jlambda=np.vstack(a_blocks)
            jgamma=np.vstack(b_blocks)
            jtrap=np.vstack(c_blocks)
            mean_signal=np.concatenate(signal_blocks)
            rms=max(float(np.sqrt(np.mean(mean_signal**2))),1e-30)
            ua,sa,rank_a=_basis(jlambda)
            ub,sb,rank_b=_basis(np.column_stack((jlambda,jgamma)))
            uc,sc,rank_c=_basis(np.column_stack((jlambda,jtrap)))
            orth_a=jtrap-ua@(ua.T@jtrap)
            gamma_after_lambda=jgamma-ua@(ua.T@jgamma)
            trap_after_lambda=orth_a
            cosine=float(np.dot(trap_after_lambda.ravel(),gamma_after_lambda.ravel())/
                         max(np.linalg.norm(trap_after_lambda)*np.linalg.norm(gamma_after_lambda),1e-30))
            # Enforce nonnegative nuisance rates.  This is a one-dimensional
            # NNLS projection after profiling out the common dimer tangent.
            gamma_per_trap=max(0.,float(np.dot(gamma_after_lambda.ravel(),trap_after_lambda.ravel())/
                                 max(np.dot(gamma_after_lambda.ravel(),gamma_after_lambda.ravel()),1e-30)))
            trap_per_gamma=max(0.,float(np.dot(trap_after_lambda.ravel(),gamma_after_lambda.ravel())/
                                 max(np.dot(trap_after_lambda.ravel(),trap_after_lambda.ravel()),1e-30)))
            orth_b=trap_after_lambda-gamma_per_trap*gamma_after_lambda
            orth_c=gamma_after_lambda-trap_per_gamma*trap_after_lambda
            for relnoise in noise_levels:
                sigma=max(float(relnoise*rms),1e-30)
                for kd in trap_rates:
                    # First-order truth shift from kd=0.  Locality is assessed
                    # through the separately reported kd*||Jd||/||S|| ratio.
                    shift=kd*jtrap.ravel()
                    mc_delta=[]
                    for seed in seeds:
                        rng=np.random.default_rng(seed)
                        noise=sigma*rng.standard_normal(shift.size)
                        observed=shift+noise
                        ra=observed-ua@(ua.T@observed)
                        gamma_hat=max(0.,float(np.dot(gamma_after_lambda.ravel(),ra)/
                                      max(np.dot(gamma_after_lambda.ravel(),gamma_after_lambda.ravel()),1e-30)))
                        rb=ra-gamma_hat*gamma_after_lambda.ravel()
                        chi_a=float(ra@ra/sigma**2)
                        chi_b=float(rb@rb/sigma**2)
                        # Gaussian AIC penalty for one additional B rate.
                        mc_delta.append((chi_b+2*4)-(chi_a+2*3))
                    records.append({"window":wname,"pixels":npix,
                        "time_set":tname,"waiting_times":list(selected_times),
                        "absorptive_only":bool(args.absorptive),
                        "relative_noise":float(relnoise),"sigma":sigma,
                        "trap_rate":float(kd),"baseline_rank":rank_a,
                        "dephasing_rank":rank_b,"trap_rank":rank_c,
                        "baseline_condition":float(sa[0]/sa[rank_a-1]) if rank_a else float("inf"),
                        "dephasing_condition":float(sb[0]/sb[rank_b-1]) if rank_b else float("inf"),
                        "trap_orthogonal_norm_per_rate":float(np.linalg.norm(orth_a)),
                        "trap_residual_after_dephasing_per_rate":float(np.linalg.norm(orth_b)),
                        "dephasing_residual_after_trap_per_rate":float(np.linalg.norm(orth_c)),
                        "residual_direction_cosine":cosine,
                        "noncentrality_A":float(np.linalg.norm(kd*orth_a)/sigma)**2,
                        "noncentrality_B":float(np.linalg.norm(kd*orth_b)/sigma)**2,
                        "relative_first_order_perturbation":float(np.linalg.norm(shift)/max(np.linalg.norm(mean_signal),1e-30)),
                        "B_aic_beats_A_fraction":float(np.mean(np.asarray(mc_delta)<0)),
                        "aic_delta_B_minus_A_mean":float(np.mean(mc_delta)),
                        "aic_delta_B_minus_A_sd":float(np.std(mc_delta,ddof=1)) if len(mc_delta)>1 else 0.})

    np.savez_compressed(out/"spectral_stress_test.npz",frequency_grid=grid,
        waiting_times=times,q0=Q0,spectra=np.array([spectra[float(t)] for t in times]),
        trap_rates=np.asarray(trap_rates),relative_noise=np.asarray(noise_levels))
    summary={"purpose":"Adversarial local tangent-space stress test; outcomes are conditional on the stated forward model and iid noise.",
      "grid":{"frequencies":grid.tolist(),"waiting_times":times.tolist(),"eta":args.eta},
      "truth_parameters":Q0.tolist(),"trap_rates":trap_rates,"relative_noise_levels":noise_levels,
      "measurement":{"noise":"iid Gaussian noise on real and imaginary samples; sigma is relative to measured-signal RMS","absorptive_only":args.absorptive,
        "equal_budget_windows":"center_3x3, low_3x3, high_3x3, and crosspeak_9 each use nine pixels; full uses nfreq squared pixels"},
      "monte_carlo_seeds":seeds,"records":records,
      "interpretation":"AIC win frequency is a local linearized diagnostic after tangent refitting, not a substitute for nonlinear likelihood fits. Large residual direction cosine means trap and dephasing remain hard to identify by microscopic identity even if both outperform A.",
      "limitations":["Finite-rate truth is first-order in kd; records report perturbation-to-signal ratio to expose breakdown of this approximation.","Noise covariance is iid and isotropic; real instrument correlations are not included.","No empirical experimental data are used."]}
    (out/"spectral_stress_test.json").write_text(json.dumps(summary,indent=2,allow_nan=True)+"\n")
    _plot(summary,out/"spectral_stress_test.png",noise_levels)
    print(json.dumps({"output_dir":str(out),"records":len(records),
       "top_high_noise_discrimination":_top_record(records),"arrays":"finite"},indent=2))
    return summary


def _top_record(records):
    eligible=[r for r in records if r["trap_rate"]>0 and r["relative_noise"]==max(x["relative_noise"] for x in records)]
    best=max(eligible,key=lambda r:r["noncentrality_B"])
    return {k:best[k] for k in ("window","time_set","trap_rate","relative_noise","noncentrality_A","noncentrality_B","B_aic_beats_A_fraction","residual_direction_cosine")}


def _plot(summary,path,noise_levels):
    records=summary["records"]
    windows=["center_3x3","low_3x3","high_3x3","crosspeak_9"]
    time_sets=["early","late","early_late","all"]
    kd=float(summary["trap_rates"][-1]); nl=max(noise_levels)
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,key,title in ((axs[0],"noncentrality_B","Trap residual after B refits (Δχ²)"),
                         (axs[1],"B_aic_beats_A_fraction","Monte Carlo P(B AIC < A AIC)")):
        mat=np.full((len(windows),len(time_sets)),np.nan)
        for r in records:
            if r["trap_rate"]==kd and r["relative_noise"]==nl and r["window"] in windows and r["time_set"] in time_sets:
                mat[windows.index(r["window"]),time_sets.index(r["time_set"])]=r[key]
        im=ax.imshow(mat,origin="lower",aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(time_sets)),time_sets,rotation=25,ha="right")
        ax.set_yticks(range(len(windows)),windows); ax.set_title(f"{title}\nkd={kd:g}, relative noise={nl:g}")
        fig.colorbar(im,ax=ax,shrink=.82)
    fig.savefig(path,dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",default="docs/benchmarks/spectral_stress_test")
    p.add_argument("--nfreq",type=int,default=5)
    p.add_argument("--omega-min",type=float,default=.90); p.add_argument("--omega-max",type=float,default=1.28)
    p.add_argument("--waiting-times",type=float,nargs="+",default=[.2,1.,4.])
    p.add_argument("--eta",type=float,default=.035)
    p.add_argument("--weak-trap",type=float,default=.0003)
    p.add_argument("--moderate-trap",type=float,default=.003)
    p.add_argument("--relative-noise",type=float,nargs="+",default=[.001,.01,.05])
    p.add_argument("--nmc",type=int,default=64)
    p.add_argument("--seed",type=int,default=1907)
    p.add_argument("--absorptive",action="store_true",help="use real-only measurements as an explicitly labeled variant")
    args=p.parse_args()
    run(args)


if __name__=="__main__": main()
