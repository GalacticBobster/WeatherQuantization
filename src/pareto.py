"""
Clean 2-panel Pareto plot per model.
  Left:  skill vs energy per forecast (captures power × time)
  Right: skill vs memory footprint

Labels use adjustText to avoid overlap.

Usage: python pareto_clean.py {dlwp|fcn}
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL    = sys.argv[1].lower() if len(sys.argv) > 1 else "dlwp"
BASE     = "/glade/derecho/scratch/ananyo/WeatherQuantization"
OUTDIR   = os.path.join(BASE, "outputs_combined")
VAR      = "z500"
LEAD_IDX = -1

# ── load hardware + skill ────────────────────────────────────
hw = pd.read_csv(os.path.join(OUTDIR, "hardware_summary.csv"))
hw = hw[hw["model"].str.lower() == MODEL]

skill = {}
for npy in glob.glob(os.path.join(BASE, "outputs_*", f"{MODEL}_bench", "*", f"{MODEL}_*_acc.npy")):
    cfg = os.path.basename(npy).replace(f"{MODEL}_","").replace("_acc.npy","")
    if cfg in skill: continue
    d = np.load(npy, allow_pickle=True).item()
    if VAR in d.get("ae", {}) and d["ae"][VAR]:
        skill[cfg] = float(np.array(d["ae"][VAR]).mean(0)[LEAD_IDX])

records = []
for _, row in hw.iterrows():
    cfg = row["config"]
    if cfg not in skill: continue
    energy_j = row["avg_power_w_mean"] * row["inference_ms_per_forecast_mean"] / 1000.0
    records.append({
        "config": cfg, "gpu": row["gpu_tag"],
        "memory": row["theoretical_size_mb_mean"],
        "energy": energy_j,
        "skill":  skill[cfg],
    })
df = pd.DataFrame(records)
if df.empty: sys.exit("No data")

# ── colors / markers ──────────────────────────────────────────
CONFIG_ORDER = ["FP32","W8A8","W8A32","W4A32","W2A32","INT8_SMOOTHQUANT","INT4_AWQ"]
COLORS  = {"FP32":"black", "W8A8":"steelblue", "W8A32":"mediumseagreen",
           "W4A32":"darkorange", "W2A32":"firebrick",
           "INT8_SMOOTHQUANT":"purple", "INT4_AWQ":"brown"}
MARKERS = {"V100":"o", "A100":"s", "H100":"^"}

# ── plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

def draw_panel(ax, xcol, xlabel):
    for cfg in CONFIG_ORDER:
        sub = df[df["config"] == cfg]
        if sub.empty: continue
        for _, row in sub.iterrows():
            ax.scatter(row[xcol], row["skill"],
                       marker=MARKERS.get(row["gpu"], "o"),
                       color=COLORS[cfg], s=200,
                       alpha=0.9, edgecolors="black", linewidths=1.3,
                       zorder=3)
    ax.axhline(0.6, color="gray", lw=0.9, ls=":", alpha=0.7, zorder=1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"ACC ({VAR})", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # smart label placement — offset outward from data center
    xmid = np.exp(np.mean(np.log(df[xcol].values)))
    ymid = df["skill"].mean()
    for _, row in df.iterrows():
        x, y = row[xcol], row["skill"]
        dx = 10 if x < xmid else -10
        dy = 10 if y < ymid else -10
        ha = "left" if dx > 0 else "right"
        va = "bottom" if dy > 0 else "top"
        ax.annotate(row["config"], (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha=ha, va=va, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="none", alpha=0.75),
                    zorder=5)

draw_panel(axes[0], "energy", "Energy per forecast (J)")
draw_panel(axes[1], "memory", "Model size (MB)")

# single external legend (both config colors and GPU markers)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
cfg_handles = [Patch(color=COLORS[c], label=c) 
               for c in CONFIG_ORDER if c in df["config"].values]
gpu_handles = [Line2D([0],[0], marker=MARKERS[g], color="black", markersize=11,
                       linestyle="none", label=g)
               for g in sorted(df["gpu"].unique())]
axes[1].legend(handles=cfg_handles + gpu_handles,
               loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=10, framealpha=0.95, title="Config / GPU",
               title_fontsize=10)

#fig.suptitle(f"{MODEL.upper()} Pareto:  skill vs efficiency  |  ACC=0.6 = dotted line",
#             fontweight="bold", fontsize=14)

out = os.path.join(OUTDIR, f"pareto_{MODEL}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
