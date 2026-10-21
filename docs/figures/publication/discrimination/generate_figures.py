#!/usr/bin/env python3
"""Rebuild publication figures from the committed spectral benchmark data.

Run from any working directory:
  python docs/figures/publication/discrimination/generate_figures.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / "docs/benchmarks/spectral_sensitivity_model_discrimination"
STRESS = ROOT / "docs/benchmarks/spectral_stress_test"

COLORS = {"A": "#34495e", "B": "#d97706", "C": "#168a75"}
PARAM_COLORS = ["#365f91", "#c4573a", "#21836b"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def _map(ax, z, extent, title, cmap="magma", vmin=None, vmax=None):
    im = ax.imshow(z, origin="lower", extent=extent, interpolation="nearest",
                   aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(r"emission frequency $\omega_3$")
    ax.set_ylabel(r"excitation frequency $\omega_1$")
    return im


def figure_spectra_and_drift(out: Path, data: dict, arrays: dict) -> None:
    freq = arrays["omega_grid"]
    waits = arrays["waiting_times"]
    spectra = arrays["spectra"]
    trap_rates = arrays["trap_rates"]
    trap_spectra = arrays["trap_truth_spectra"]
    # Baseline and moderate-trap complex response at the earliest waiting time.
    tix = int(np.argmin(np.abs(waits - 0.2)))
    kix = int(np.argmin(np.abs(trap_rates - 0.003)))
    fig = plt.figure(figsize=(11.2, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    vmax = max(np.abs(spectra[tix]).max(), np.abs(trap_spectra[kix, tix]).max())
    i1 = _map(ax1, np.abs(spectra[tix]), [freq[0], freq[-1], freq[0], freq[-1]],
              r"A  Dimer baseline, $t_2=0.2$")
    _map(ax2, np.abs(trap_spectra[kix, tix]), [freq[0], freq[-1], freq[0], freq[-1]],
         r"B  Trap truth, $k_d=0.003$")
    i1.set_clim(0, vmax)
    # Consistent scale for first two spectrum panels.
    for ax in [ax1, ax2]:
        for im in ax.images: im.set_clim(0, vmax)
    cb = fig.colorbar(i1, ax=[ax1, ax2], shrink=.86, pad=.02)
    cb.set_label(r"$|S_{RP+NRP}|$ (response units)")

    # Explain that the complex observable is preserved; phase displayed as hue is not needed here.
    baseline = data["baseline_parameters"]
    ax3.axis("off")
    ax3.text(.03, .91, "COMPLEX OBSERVABLE", transform=ax3.transAxes,
             fontsize=9, fontweight="bold", color="#273746")
    ax3.text(.03, .76, "Displayed map: modulus of RP + NRP\nFits and information calculations retain\nstacked real and imaginary components.",
             transform=ax3.transAxes, va="top", linespacing=1.5)
    ax3.text(.03, .43, "Baseline parameters", transform=ax3.transAxes, fontweight="bold")
    ax3.text(.03, .31, rf"$J_{{12}}={baseline['J12']:.3f}$,  $k_{{12}}={baseline['k12']:.3f}$," + "\n" + rf"$\gamma_\phi={baseline['gamma_phi']:.3f}$",
             transform=ax3.transAxes, va="top", linespacing=1.5)
    ax3.text(.03, .08, "Darker sink has no optical dipole or return path.", transform=ax3.transAxes,
             fontsize=8, color="#505b66")

    # Apparent parameter drift under independent fits to each time slice.
    labels = [(r"$J_{12}$ (energy)", 0), (r"$k_{12}$ (time$^{-1}$)", 1),
              (r"$\gamma_\phi$ (time$^{-1}$)", 2)]
    traps = [0.0003, 0.003, 0.012]
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    base = np.array([baseline["J12"], baseline["k12"], baseline["gamma_phi"]])
    for pidx, (label, j) in enumerate(labels):
        ax = axes[pidx]
        ax.axhline(base[j], color="#222222", lw=1.1, ls="--", label="truth" if pidx == 0 else None)
        for ti, kd in enumerate(traps):
            color = ["#4c78a8", "#e07b39", "#9a5aa0"][ti]
            xs=[]; ys=[]; lo=[]; hi=[]
            for t in waits:
                key=f"kd={kd:g},t={t:g}"
                rec=data["single_time_baseline_fits"][key]
                xs.append(t); ys.append(rec["x"][j]); lo.append(rec["ci95"][j][0]); hi.append(rec["ci95"][j][1])
            ax.errorbar(xs, ys, yerr=[np.array(ys)-lo, hi-np.array(ys)], marker="o", ms=4,
                        lw=1.2, capsize=2, color=color, label=rf"$k_d={kd:g}$")
        ax.set_title(chr(ord("C")+pidx)+"  "+label, loc="left", fontweight="semibold")
        ax.set_xlabel(r"waiting time $t_2$")
        if pidx == 0: ax.set_ylabel("reduced-model estimate")
        ax.grid(axis="y", alpha=.22)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle("Spectral response and apparent parameter drift under a hidden sink", fontweight="bold")
    fig.savefig(out / "figure_hidden_channel_model_discrimination.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_hidden_channel_model_discrimination.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def figure_model_residuals(out: Path, data: dict) -> None:
    fits=data["joint_model_comparison"]["fits"]
    waits=[0.2, 1.0, 4.0]
    fig, axes=plt.subplots(3,3,figsize=(9.6,8.6),layout="constrained")
    mats=[]
    for model in "ABC":
        for t in waits:
            r=fits[model]["residual_maps"][str(t)]
            mats.append(np.hypot(np.asarray(r["real"]),np.asarray(r["imag"])))
    finite=np.concatenate([m.ravel() for m in mats if m.size])
    vmax=np.quantile(finite, .995)
    vmax=max(vmax,1e-12)
    for ri,model in enumerate("ABC"):
        for ci,t in enumerate(waits):
            rec=fits[model]["residual_maps"][str(t)]
            mag=np.hypot(np.asarray(rec["real"]),np.asarray(rec["imag"]))
            ax=axes[ri,ci]
            # A small positive floor keeps the exact-model residual visible on a shared log scale.
            im=ax.imshow(np.log10(np.maximum(mag,1e-15)),origin="lower",extent=[.9,1.28,.9,1.28],
                         cmap="viridis",vmin=-15,vmax=np.log10(vmax),aspect="equal")
            model_name={"A":"baseline dimer","B":"extra dephasing","C":"explicit trap"}[model]
            ax.set_title(f"{model}: {model_name}  |  $t_2={t:g}$",loc="left",fontweight="semibold",color=COLORS[model])
            if ri==2: ax.set_xlabel(r"emission frequency $\omega_3$")
            if ci==0: ax.set_ylabel(r"excitation frequency $\omega_1$")
            ax.tick_params(length=2)
    cb=fig.colorbar(im,ax=axes,shrink=.82,pad=.02)
    cb.set_label(r"$\log_{10}|S_{model}-S_{truth}|$ (response units)")
    fig.suptitle("Complex residual structure for joint fits to trap-generated spectra\n(C is the generating model; its residual is at numerical precision)",fontweight="bold")
    fig.savefig(out/"figure2_ABC_complex_residuals.pdf",bbox_inches="tight")
    fig.savefig(out/"figure2_ABC_complex_residuals.png",dpi=360,bbox_inches="tight")
    plt.close(fig)

    # Compact metrics panel: joint-fit weighted SSE and held-out waiting-time prediction errors.
    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(9.4,3.7),layout="constrained")
    models=list("ABC")
    costs=[fits[m]["cost"] for m in models]
    ax1.bar(models,costs,color=[COLORS[m] for m in models],width=.62)
    ax1.set_yscale("log"); ax1.set_ylabel("joint weighted SSE")
    ax1.set_title("A  Training residual",loc="left",fontweight="semibold")
    ax1.grid(axis="y",which="both",alpha=.2)
    # Exclude the joint-fit training waits (0.2, 1.0, 4.0); only display the
    # intervening predictions as held-out data.
    train_waits={0.2,1.0,4.0}
    htimes=[t for t in fits["A"]["heldout"] if float(t) not in train_waits]
    for m in models:
        vals=[max(fits[m]["heldout"][t],1e-18) for t in htimes]
        ax2.plot(np.arange(len(htimes)),vals,marker="o",ms=4,lw=1.5,color=COLORS[m],label=m)
    ax2.set_xticks(np.arange(len(htimes)),[f"{float(t):g}" for t in htimes])
    ax2.set_yscale("log"); ax2.set_xlabel(r"held-out $t_2$"); ax2.set_ylabel("held-out spectral error")
    ax2.set_title("B  Held-out prediction (training waits excluded)",loc="left",fontweight="semibold")
    ax2.legend(frameon=False,title="model"); ax2.grid(axis="y",which="both",alpha=.2)
    fig.suptitle("Fit quality is informative only alongside predictive checks",fontweight="bold")
    fig.savefig(out/"figure3_fit_and_heldout_metrics.pdf",bbox_inches="tight")
    fig.savefig(out/"figure3_fit_and_heldout_metrics.png",dpi=360,bbox_inches="tight")
    plt.close(fig)


def figure_stress_test(out: Path, stress: dict) -> None:
    records=stress["records"]
    windows=["center_3x3","low_3x3","high_3x3","crosspeak_9","full"]
    time_sets=["early","early_late","late"]
    noises=sorted(stress["relative_noise_levels"])
    traps=sorted(stress["trap_rates"])
    # Show model-selection frequency for the moderate trap rate, preserving equal budgets by window.
    target=traps[-1]
    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(11.2,5.2),gridspec_kw={"width_ratios":[1.6,1]},layout="constrained")
    rowlabels=[]; mat=[]
    for noise in noises:
        for ts in time_sets:
            rowlabels.append(f"{ts.replace('_','–')}\n{noise:.1%}")
            row=[]
            for win in windows:
                rec=next(r for r in records if r["window"]==win and r["time_set"]==ts
                         and r["relative_noise"]==noise and r["trap_rate"]==target)
                row.append(rec["B_aic_beats_A_fraction"])
            mat.append(row)
    im=ax1.imshow(mat,origin="upper",aspect="auto",cmap="Blues",vmin=0,vmax=1)
    window_labels=["center\n9 px","low\n9 px","high\n9 px","crosspeak\n9 px","full\n25 px"]
    ax1.set_xticks(range(len(windows)),window_labels)
    ax1.set_yticks(range(len(rowlabels)),rowlabels)
    ax1.set_xlabel("measured spectral window (pixel budget shown in label)")
    ax1.set_title(rf"A  Fraction of seeds where B beats A ($k_d={target:g}$)",loc="left",fontweight="semibold")
    for i in range(len(mat)):
        for j in range(len(windows)):
            ax1.text(j,i,f"{mat[i][j]:.2f}",ha="center",va="center",
                     color="white" if mat[i][j]>.58 else "#243447",fontsize=7)
    fig.colorbar(im,ax=ax1,shrink=.78,label="Monte Carlo selection fraction")

    # Local tangent directions are nearly collinear across candidate time sets.
    cosine=[]; contrast=[]
    for ts in time_sets:
        recs=[r for r in records if r["window"]=="full" and r["time_set"]==ts
              and r["relative_noise"]==noises[0] and r["trap_rate"]==target]
        cosine.append(recs[0]["residual_direction_cosine"])
        design=next(x for x in json.load(open(BENCH/"spectral_sensitivity_benchmark.json"))["trap_vs_dephasing_design_candidates"]
                    if [float(v) for v in x["times"]] == {"early":[.2],"early_late":[.2,4.],"late":[4.]}[ts])
        contrast.append(design["trap_dephasing_contrast_score"])
    xx=np.arange(len(time_sets))
    ax2.plot(xx,cosine,"o-",lw=1.8,color="#8a4f9e",label="direction cosine")
    ax2.set_ylim(.95,1.005); ax2.set_ylabel("trap/dephasing direction cosine")
    ax2.set_xticks(xx,[s.replace("_"," + ") for s in time_sets])
    ax2.set_xlabel("waiting-time set")
    ax2.set_title("B  Mechanism directions remain aligned",loc="left",fontweight="semibold")
    ax2.grid(axis="y",alpha=.22)
    ax2.text(.04,.05,"Near 1 means local spectral\nperturbations are hard to separate.",transform=ax2.transAxes,fontsize=8,color="#505b66")
    fig.suptitle("Adversarial stress test: discrimination depends on noise, time, and window",fontweight="bold")
    fig.savefig(out/"figure_stress_design_robustness.pdf",bbox_inches="tight")
    fig.savefig(out/"figure_stress_design_robustness.png",dpi=360,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    data=json.loads((BENCH/"spectral_sensitivity_benchmark.json").read_text())
    stress=json.loads((STRESS/"spectral_stress_test.json").read_text())
    z=np.load(BENCH/"spectral_sensitivity_benchmark.npz")
    arrays={k:z[k] for k in z.files}
    figure_spectra_and_drift(out,data,arrays)
    figure_model_residuals(out,data)
    figure_stress_test(out,stress)
    print(f"Wrote publication figures to {out}")


if __name__=="__main__": main()
