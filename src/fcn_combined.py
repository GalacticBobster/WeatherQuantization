import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR_BASE = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs/fcn_results"
OUTDIR_PLOT = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs/fcn_combined"
VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]

EXPERIMENTS = {
    "FP32":             "black",
    "W8A8":             "steelblue",
    "W8A32":            "mediumseagreen",
    "W4A32":            "darkorange",
    "W2A32":            "firebrick",
    "INT8_SMOOTHQUANT": "purple",
    "INT4_AWQ":         "brown",
}

os.makedirs(OUTDIR_PLOT, exist_ok=True)

results, leads_ref = {}, None
for exp in EXPERIMENTS:
    slug = f"fcn_{exp}"
    d    = OUTDIR_BASE
    try:
        results[exp] = {
            "gfs":  np.load(os.path.join(d, f"{slug}_rmse_vs_gfs.npy"),  allow_pickle=True).item(),
            "fp32": np.load(os.path.join(d, f"{slug}_rmse_vs_fp32.npy"), allow_pickle=True).item(),
            "gfs_fp32": np.load(os.path.join(d, f"{slug}_fp32_vs_gfs.npy"), allow_pickle=True).item(),
        }
        if leads_ref is None:
            leads_ref = np.load(os.path.join(d, f"{slug}_leads.npy"))
        print(f"Loaded: {exp}")
    except FileNotFoundError:
        print(f"Missing: {exp} — skipping")

if not results:
    print("No results found."); exit(1)

MODEL_VARS = [v for v in VARIABLES if any(v in results[e]["gfs"] for e in results)]
agg        = lambda lst: (np.stack(lst).mean(0), np.stack(lst).std(0))
ncols      = min(4, len(MODEL_VARS))
nrows      = int(np.ceil(len(MODEL_VARS) / ncols))

for metric, ylabel, fname in [
    ("gfs",  "Lat-wtd RMSE vs GFS",  "fcn_combined_rmse_vs_gfs.png"),
    ("fp32", "Lat-wtd RMSE vs FP32", "fcn_combined_rmse_vs_fp32.png"),
]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax, v in zip(axes, MODEL_VARS):
        for exp, col in EXPERIMENTS.items():
            if metric == "fp32" and exp == "FP32": continue
            if exp not in results or v not in results[exp][metric]: continue
            m, s = agg(results[exp][metric][v])
            ls   = "-" if exp == "FP32" else "--"
            ax.plot(leads_ref, m, color=col, lw=2, ls=ls, label=exp)
            ax.fill_between(leads_ref, m-s, m+s, color=col, alpha=0.10)
        ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    for ax in axes[len(MODEL_VARS):]: ax.set_visible(False)
    fig.suptitle(f"FCN {ylabel} — all configs | mean ±1σ", fontweight="bold")
    fig.savefig(os.path.join(OUTDIR_PLOT, fname), dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

# FP32 self-consistency
if "FP32" in results:
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes3 = np.array(axes3).flatten()
    for ax, v in zip(axes3, MODEL_VARS):
        if v not in results["FP32"]["fp32"]: ax.set_visible(False); continue
        m, s = agg(results["FP32"]["fp32"][v])
        ax.plot(leads_ref, m, color="black", lw=2)
        ax.fill_between(leads_ref, m-s, m+s, color="black", alpha=0.15)
        ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("RMSE FP32 vs FP32"); ax.grid(True, alpha=0.3)
    for ax in axes3[len(MODEL_VARS):]: ax.set_visible(False)
    fig3.suptitle("FCN FP32 self-consistency — numerical noise floor", fontweight="bold")
    fig3.savefig(os.path.join(OUTDIR_PLOT, "fcn_fp32_self_consistency.png"), dpi=150)
    plt.close(fig3)
    print("Saved: fcn_fp32_self_consistency.png")

print("Done.")
