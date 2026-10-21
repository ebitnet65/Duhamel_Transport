#!/usr/bin/env python3
"""Summarize saved dark-state trap-rate transport fits and residual spectra."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

PARAMETERS=(r"$J_{12}$",r"$k_{12}$",r"$\gamma_\phi$")
UNITS=("eV",r"time$^{-1}$",r"time$^{-1}$")


def _load(folder: Path):
    json_path=folder/"dark_state_transport_comparison.json"
    npz_path=folder/"dark_state_transport_comparison.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"Missing comparison JSON/NPZ in {folder}")
    meta=json.loads(json_path.read_text())
    with np.load(npz_path) as src:
        data={key:src[key].copy() for key in src.files}
    numeric=[v for v in data.values() if np.issubdtype(v.dtype,np.number)]
    if not all(np.all(np.isfinite(v)) for v in numeric):
        raise ValueError(f"Nonfinite numerical output in {npz_path}")
    return meta,data


def _relative_error(pred,truth):
    return float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-30))


def _collect(baseline: Path, sweep_root: Path):
    inputs=[baseline]
    if sweep_root.exists():
        inputs.extend(sorted(p for p in sweep_root.glob("kd_*") if p.is_dir()))
    records=[]
    for folder in inputs:
        meta,data=_load(folder)
        kd=float(meta["trap_rate"])
        times=np.asarray(data["waiting_times"],float)
        truth=np.asarray(data["truth_spectra"])
        joint=np.asarray(data["joint_transported_spectra"])
        independent=np.asarray(data["per_time_transported_spectra"])
        joint_rel=[_relative_error(p,y) for p,y in zip(joint,truth,strict=True)]
        per_rel=[_relative_error(p,y) for p,y in zip(independent,truth,strict=True)]
        records.append({"kd":kd,"path":str(folder),"times":times.tolist(),
                        "truth_parameters":np.asarray(data["truth_parameters"],float).tolist(),
                        "joint_parameters":np.asarray(data["joint_fit_parameters"],float).tolist(),
                        "per_time_parameters":np.asarray(data["per_time_fit_parameters"],float).tolist(),
                        "joint_relative_complex_l2":joint_rel,
                        "per_time_relative_complex_l2":per_rel,
                        "joint_fit_success":bool(meta["joint_fit"]["success"]),
                        "joint_fit_cost":float(meta["joint_fit"]["cost"]),
                        "waiting_times":times.tolist(),"truth_spectra":truth,
                        "joint_spectra":joint,"per_time_spectra":independent,
                        "omega_grid":np.asarray(data["omega_grid"],float),
                        "normalization":float(data["shared_truth_l2_normalization"])})
    records.sort(key=lambda x:x["kd"])
    if not records: raise ValueError("No baseline or trap-sweep result directories found")
    return records


def _save_trend_figure(records,out):
    kd=np.asarray([r["kd"] for r in records])
    truth=np.asarray(records[0]["truth_parameters"])
    times=np.asarray(records[0]["times"])
    fig,axs=plt.subplots(1,4,figsize=(15.5,4.1),constrained_layout=True)
    colors=["#1b9e77","#d95f02","#7570b3"]
    for p,ax in enumerate(axs[:3]):
        joint=np.asarray([r["joint_parameters"][p] for r in records])
        independent=np.asarray([r["per_time_parameters"] for r in records])[:,:,p]
        for ti,t in enumerate(times):
            ax.plot(kd,independent[:,ti],"o--",color=colors[ti%len(colors)],alpha=.62,
                    ms=4,label=rf"independent $t_2={t:g}$" if p==0 else None)
        ax.plot(kd,joint,"ko-",lw=1.8,ms=5,label="joint fit" if p==0 else None)
        ax.axhline(truth[p],color="#555555",ls=":",lw=1.4,label="truth control" if p==0 else None)
        ax.set_xscale("log"); ax.set_xlabel(r"dark-trap rate $k_d$")
        ax.set_ylabel(f"{PARAMETERS[p]} ({UNITS[p]})")
        ax.grid(alpha=.25)
        ax.set_title(chr(65+p),loc="left",fontweight="bold")
    axs[0].legend(frameon=False,fontsize=7)
    ax=axs[3]
    for ti,t in enumerate(times):
        jointerr=[r["joint_relative_complex_l2"][ti] for r in records]
        pererr=[r["per_time_relative_complex_l2"][ti] for r in records]
        ax.plot(kd,jointerr,"o-",color=colors[ti%len(colors)],label=rf"joint, $t_2={t:g}$")
        ax.plot(kd,pererr,"x--",color=colors[ti%len(colors)],alpha=.7,label=rf"separate, $t_2={t:g}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"dark-trap rate $k_d$")
    ax.set_ylabel("relative complex-spectrum L2 residual")
    ax.grid(alpha=.25); ax.set_title("D",loc="left",fontweight="bold")
    ax.legend(frameon=False,fontsize=6.5,ncol=2)
    fig.suptitle("Where the sink-free transported dimer stops absorbing dark-state loss",
                 fontsize=14,fontweight="bold")
    fig.savefig(out/"figure_dark_state_trap_rate_sweep.pdf",bbox_inches="tight")
    fig.savefig(out/"figure_dark_state_trap_rate_sweep.png",dpi=600,bbox_inches="tight")
    plt.close(fig)


def _save_high_trap_spectra(record,out):
    times=np.asarray(record["times"])
    ti=int(np.argmax(times))
    grid=record["omega_grid"]
    truth=record["truth_spectra"][ti]/record["normalization"]
    fit=record["joint_spectra"][ti]/record["normalization"]
    residual=fit-truth
    amplitude_limit=max(float(np.max(np.abs(truth))),float(np.max(np.abs(fit))),1e-30)
    limit=max(float(np.max(np.abs(residual.real))),float(np.max(np.abs(residual.imag))),1e-30)
    fig,axs=plt.subplots(1,4,figsize=(14.6,3.7),constrained_layout=True)
    panels=[("Truth $|S|$",np.abs(truth),"magma",None),
            ("Best transported $|S|$",np.abs(fit),"magma",None),
            ("Residual Re",residual.real,"RdBu_r",TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)),
            ("Residual Im",residual.imag,"RdBu_r",TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))]
    for ax,(title,z,cmap,norm) in zip(axs,panels,strict=True):
        if norm is None:
            im=ax.imshow(z,origin="lower",extent=[grid[0],grid[-1],grid[0],grid[-1]],
                         aspect="equal",cmap=cmap,vmin=0,vmax=amplitude_limit)
        else:
            im=ax.imshow(z,origin="lower",extent=[grid[0],grid[-1],grid[0],grid[-1]],
                         aspect="equal",cmap=cmap,norm=norm)
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xlabel(r"$\omega_1$ (eV)"); ax.set_ylabel(r"$\omega_3$ (eV)")
        fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle(rf"Strongest completed trap case: $k_d={record['kd']:g}$, $t_2={times[ti]:g}$",
                 fontsize=13,fontweight="bold")
    fig.savefig(out/"figure_dark_state_strong_trap_residuals.pdf",bbox_inches="tight")
    fig.savefig(out/"figure_dark_state_strong_trap_residuals.png",dpi=600,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir",type=Path,default=Path("publication_artifacts/dark_state_transport"))
    p.add_argument("--sweep-root",type=Path,default=Path("publication_artifacts/dark_state_transport/trap_sweep"))
    p.add_argument("--output-dir",type=Path,default=None)
    a=p.parse_args()
    output=a.output_dir or a.sweep_root
    output.mkdir(parents=True,exist_ok=True)
    records=_collect(a.baseline_dir,a.sweep_root)
    _save_trend_figure(records,output)
    _save_high_trap_spectra(records[-1],output)
    summary=[]
    for r in records:
        summary.append({k:v for k,v in r.items() if k not in {"times","truth_spectra","joint_spectra","per_time_spectra","omega_grid"}})
    (output/"dark_state_trap_sweep_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (output/"dark_state_trap_sweep_metrics.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["kd","fit","waiting_time","J12","k12","gamma_phi","relative_complex_l2"])
        for r in records:
            for kind,param_key,error_key in (("joint","joint_parameters","joint_relative_complex_l2"),
                                             ("independent","per_time_parameters","per_time_relative_complex_l2")):
                params=np.asarray(r[param_key])
                if kind=="joint": params=np.tile(params,(len(r["times"]),1))
                for t,q,e in zip(r["times"],params,r[error_key],strict=True):
                    writer.writerow([r["kd"],kind,t,*q,e])
    print(json.dumps({"output_dir":str(output),"trap_rates":[r["kd"] for r in records],
                      "joint_parameters":[r["joint_parameters"] for r in records],
                      "joint_relative_complex_l2":[r["joint_relative_complex_l2"] for r in records]},indent=2))


if __name__=="__main__": main()
