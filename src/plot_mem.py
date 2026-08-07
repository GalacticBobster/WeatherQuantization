"""
Memory reduction — DLWP + FCN back-to-back, vertically stacked bar plot.
Reads hardware_metrics.csv from all outputs_* dirs (V100, A100, etc.)

Usage:  python plot_memory.py
Output: outputs_combined/memory_reduction.png
"""
import os, glob, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = "/glade/derecho/scratch/ananyo/WeatherQuantization"
OUTDIR = os.path.join(BASE, "outputs_combined")
os.makedirs(OUTDIR, exist_ok=True)

CONFIG_ORDER = ["FP32","W8A8","W8A32","W4A32","W2A32","INT8_SMOOTHQUANT","INT4_AWQ"]
GPU_ORDER    = ["V100", "A100", "H100"]
GPU_COLORS   = {"V100": "#4c72b0", "A100": "#dd8452", "H100": "#55a868"}

# ── read all CSVs ─────────────────────────────────────────────
pattern = os.path.join(BASE, "outputs_*", "*_bench", "*", "hardware_metrics.csv")
files   = sorted(glob.glob(pattern))
rows = []
for path in files:
    parts   = path.split(os.sep)
    gpu_tag = next(p for p in parts if p.startswith("outputs_")).replace("outputs_", "")
    with open(path) as f:
        for r in csv.DictReader(f):
            r["gpu_tag"] = gpu_tag
            rows.append(r)

if not rows:
    raise SystemExit(f"No CSV data found in {pattern}")

df = pd.DataFrame(rows)
for c in ["theoretical_size_mb","fp32_size_mb"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace("x",""), errors="coerce")

# take minimum size per (model, config) — size is deterministic, same across runs
agg = df.groupby(["model","config"]).agg({
    "theoretical_size_mb": "min",
    "fp32_size_mb":        "min",
}).reset_index()

# ── plot: 2 vertically stacked panels, one per model ─────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

for ax, model in zip(axes, ["DLWP", "FCN"]):
    sub = agg[agg["model"].str.upper() == model]
    configs = [c for c in CONFIG_ORDER if c in set(sub["config"])]

    sizes = []
    ratios = []
    fp32_mb = None
    for cfg in configs:
        row = sub[sub["config"] == cfg]
        if len(row) == 0:
            sizes.append(np.nan); ratios.append(np.nan); continue
        mb   = row["theoretical_size_mb"].values[0]
        f32m = row["fp32_size_mb"].values[0]
        if fp32_mb is None: fp32_mb = f32m
        sizes.append(mb)
        ratios.append(f32m / mb if mb > 0 else 1.0)

    x = np.arange(len(configs))
    bars = ax.bar(x, sizes, width=0.6, color="#4c72b0",
                    edgecolor="black", linewidth=1.2)

    # annotate compression ratio above each bar
    ymax = max(sizes) * 1.15
    for xi, (bar, r, s) in enumerate(zip(bars, ratios, sizes)):
        if np.isnan(s): continue
        ax.text(bar.get_x() + bar.get_width()/2, s + ymax*0.01,
                f"{s:.1f} MB\n({r:.1f}×)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=14)
    ax.set_ylabel("Model size (MB)", fontsize=14)
    ax.set_title(f"{model} Memory Footprint",
                  fontweight="bold", fontsize=15)
    ax.set_ylim([0, ymax])
    ax.grid(True, alpha=0.3, axis="y")

    # dashed FP32 reference line
    if fp32_mb:
        ax.axhline(fp32_mb, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        #ax.text(len(configs)-0.2, fp32_mb, f"FP32 baseline ({fp32_mb:.1f} MB)",
        #        ha="center", va="bottom", fontsize=8, color="gray", style="italic")

#fig.suptitle("Memory reduction across PTQ configurations",
#              fontweight="bold", fontsize=14)

out = os.path.join(OUTDIR, "memory_reduction.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
