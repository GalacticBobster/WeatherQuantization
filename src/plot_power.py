"""
Power consumption — DLWP + FCN back-to-back, vertically stacked bar plot.
Includes error bars (min/max range across runs) per GPU.

Usage:  python plot_power.py
Output: outputs_combined/power_consumption.png
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
df["avg_power_w"] = pd.to_numeric(df["avg_power_w"], errors="coerce")
df = df.dropna(subset=["avg_power_w"])

# aggregate: mean + min + max per (model, gpu, config)
agg = df.groupby(["model","gpu_tag","config"]).agg({
    "avg_power_w": ["mean", "min", "max", "count"],
}).reset_index()
agg.columns = ["model","gpu_tag","config","power_mean","power_min","power_max","count"]

# ── plot: 2 vertically stacked panels, one per model ─────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

for ax, model in zip(axes, ["DLWP", "FCN"]):
    sub = agg[agg["model"].str.upper() == model]
    configs = [c for c in CONFIG_ORDER if c in set(sub["config"])]
    gpus    = [g for g in GPU_ORDER if g in set(sub["gpu_tag"])]

    x = np.arange(len(configs))
    width = 0.8 / max(len(gpus), 1)

    max_top = 0
    for i, gpu in enumerate(gpus):
        means, err_low, err_high = [], [], []
        for cfg in configs:
            row = sub[(sub["gpu_tag"] == gpu) & (sub["config"] == cfg)]
            if len(row) == 0:
                means.append(np.nan); err_low.append(0); err_high.append(0); continue
            m  = row["power_mean"].values[0]
            lo = row["power_min"].values[0]
            hi = row["power_max"].values[0]
            means.append(m)
            err_low.append(m - lo)
            err_high.append(hi - m)
        pos = x + (i - (len(gpus)-1)/2) * width
        ax.bar(pos, means, width, yerr=[err_low, err_high],
                color=GPU_COLORS.get(gpu, "gray"), edgecolor="black",
                linewidth=1.0, capsize=4, label=gpu, alpha=0.9)
        finite_top = np.nanmax(np.array(means) + np.array(err_high))
        if not np.isnan(finite_top):
            max_top = max(max_top, finite_top)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=12)
    ax.set_ylabel("GPU power (W)", fontsize=14)
    ax.set_title(f"{model}",
                  fontweight="bold", fontsize=14)
    if max_top > 0: ax.set_ylim([0, max_top * 1.15])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, framealpha=0.9, title="GPU", loc="upper right")

#fig.suptitle("Power consumption across PTQ configurations "
#              "(error bars: min–max across runs)",
#              fontweight="bold", fontsize=14)

out = os.path.join(OUTDIR, "power_consumption.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
