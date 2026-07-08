"""
Aggregate hardware metrics across all GPU types and configs.
Auto-discovers outputs_*/​<model>_bench/<config>/hardware_metrics.csv
"""
import os, sys, csv, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = "/glade/derecho/scratch/ananyo/WeatherQuantization"
OUTDIR = os.path.join(BASE, "outputs_combined")
os.makedirs(OUTDIR, exist_ok=True)

pattern = os.path.join(BASE, "outputs_*", "*_bench", "*", "hardware_metrics.csv")
files   = sorted(glob.glob(pattern))
print(f"Found {len(files)} CSV files")

rows = []
for path in files:
    parts   = path.split(os.sep)
    gpu_tag = next(p for p in parts if p.startswith("outputs_")).replace("outputs_", "")
    with open(path) as f:
        for r in csv.DictReader(f):
            r["gpu_tag"] = gpu_tag
            rows.append(r)

if not rows: sys.exit("No data rows")
df = pd.DataFrame(rows)
for c in ["theoretical_size_mb","fp32_size_mb","inference_ms_per_forecast",
          "avg_power_w","params_M","flops_step_G","flops_forecast_G"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("x",""), errors="coerce")

df.to_csv(os.path.join(OUTDIR, "hardware_combined.csv"), index=False)
print(f"Combined → {len(df)} rows")

# aggregate multiple runs per (model,gpu,config)
agg = df.groupby(["model","gpu_tag","config"]).agg({
    "theoretical_size_mb":       ["mean","std","count"],
    "inference_ms_per_forecast": ["mean","std"],
    "avg_power_w":               ["mean","std"],
    "params_M":                  "mean",
    "flops_step_G":              "mean",
    "flops_forecast_G":          "mean",
})
agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
agg = agg.reset_index()
agg.to_csv(os.path.join(OUTDIR, "hardware_summary.csv"), index=False)
print(f"Summary  → {len(agg)} unique combos")
print(agg.to_string(index=False))

# ── bar plots per model ───────────────────────────────────────
CONFIG_ORDER = ["FP32","W8A8","W8A32","W4A32","W2A32","INT8_SMOOTHQUANT","INT4_AWQ"]

for model in agg["model"].unique():
    sub     = agg[agg["model"] == model]
    gpus    = sorted(sub["gpu_tag"].unique())
    configs = [c for c in CONFIG_ORDER if c in set(sub["config"])]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    metrics = [
        ("theoretical_size_mb_mean",       "Model size (MB)",         "Memory"),
        ("inference_ms_per_forecast_mean", "Inference (ms/forecast)", "Inference time"),
        ("avg_power_w_mean",               "GPU power (W)",           "Power draw"),
    ]
    
    x = np.arange(len(configs))
    width = 0.8 / max(len(gpus), 1)
    
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for i, gpu in enumerate(gpus):
            vals, errs = [], []
            for cfg in configs:
                row = sub[(sub["gpu_tag"] == gpu) & (sub["config"] == cfg)]
                vals.append(row[col].values[0] if len(row) else np.nan)
                err_col = col.replace("_mean","_std")
                errs.append(row[err_col].values[0] if err_col in sub.columns and len(row) else 0)
            ax.bar(x + i*width, vals, width, yerr=errs, label=gpu, capsize=3)
        ax.set_xticks(x + width*(len(gpus)-1)/2)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(f"{model} hardware metrics across GPUs", fontweight="bold", fontsize=13)
    out = os.path.join(OUTDIR, f"hardware_{model.lower()}_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot     → {out}")
