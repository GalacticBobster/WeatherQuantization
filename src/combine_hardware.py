"""
Aggregate hardware metrics across all GPU types and configs.
Auto-discovers outputs_*/<model>_bench/<config>/hardware_metrics.csv
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
            r["_src"]    = path
            rows.append(r)

if not rows: sys.exit("No data rows")
df = pd.DataFrame(rows)

NUMERIC = ["theoretical_size_mb", "fp32_size_mb", "params_M",
           "inference_ms_per_forecast", "inference_ms_std",
           "idle_power_w", "mean_power_w", "power_std_w", "peak_power_w",
           "net_power_w", "energy_j_per_forecast", "n_power_samples"]
for c in NUMERIC:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("x", ""), errors="coerce")

# ── schema guard ──────────────────────────────────────────────
# Rows written before the pynvml change carry avg_power_w and the FLOPs columns
# instead of mean_power_w / energy_j_per_forecast. Concatenating the two fills
# the missing side with NaN and the aggregation below would silently average
# over whichever subset happens to have each column.
if "avg_power_w" in df.columns:
    stale = df["mean_power_w"].isna() if "mean_power_w" in df.columns else pd.Series(True, index=df.index)
    n_stale = int(stale.sum())
    if n_stale:
        print(f"\n!! {n_stale} of {len(df)} rows use the OLD schema "
              f"(avg_power_w, 2-sample instantaneous power).")
        print("   These are not comparable to the 20 Hz load averages. Sources:")
        for p in sorted(df.loc[stale, "_src"].unique()):
            print(f"     {p}")
        print("   Dropping them. Re-run the benchmark pass for these configs.\n")
        df = df[~stale].copy()
        if df.empty:
            sys.exit("All rows were stale — re-run the hardware benchmark.")

for col in ["avg_power_w", "flops_step_G", "flops_forecast_G"]:
    df = df.drop(columns=col, errors="ignore")

# ── sampler sanity check ──────────────────────────────────────
if "n_power_samples" in df.columns:
    dead = df[df["n_power_samples"].fillna(0) < 10]
    if len(dead):
        print(f"!! {len(dead)} rows have <10 power samples — pynvml likely failed "
              f"on those nodes; their power/energy values are unreliable:")
        print(dead[["model", "gpu_tag", "config", "n_power_samples"]].to_string(index=False))
        print()

df.drop(columns="_src", errors="ignore").to_csv(
    os.path.join(OUTDIR, "hardware_combined.csv"), index=False)
print(f"Combined → {len(df)} rows")

# ── aggregate multiple runs per (model, gpu, config) ──────────
AGG_SPEC = {
    "theoretical_size_mb":       ["mean", "std", "count"],
    "fp32_size_mb":              ["mean"],
    "inference_ms_per_forecast": ["mean", "std"],
    "mean_power_w":              ["mean", "std"],
    "energy_j_per_forecast":     ["mean", "std"],
    "idle_power_w":              ["mean"],
    "peak_power_w":              ["mean"],
    "net_power_w":               ["mean"],
    "params_M":                  ["mean"],
}
spec = {k: v for k, v in AGG_SPEC.items() if k in df.columns}
missing = [k for k in AGG_SPEC if k not in df.columns]
if missing:
    print(f"note: columns absent from all CSVs, skipped → {missing}")

agg = df.groupby(["model", "gpu_tag", "config"]).agg(spec)
agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
agg = agg.reset_index()
agg.to_csv(os.path.join(OUTDIR, "hardware_summary.csv"), index=False)
print(f"Summary  → {len(agg)} unique combos")
print(agg.to_string(index=False))

# ── bar plots per model ───────────────────────────────────────
CONFIG_ORDER = ["FP32", "W8A8", "W8A32", "W4A32", "W2A32",
                "INT8_SMOOTHQUANT", "INT4_AWQ"]

METRICS = [
    # ("theoretical_size_mb_mean",     "Model size (MB)",         "Memory"),
    ("inference_ms_per_forecast_mean", "Inference (ms/forecast)", "Inference time"),
    ("mean_power_w_mean",              "GPU power (W)",           "Power draw"),
    ("energy_j_per_forecast_mean",     "Energy (J/forecast)",     "Energy per forecast"),
]

for model in agg["model"].unique():
    sub     = agg[agg["model"] == model]
    gpus    = sorted(sub["gpu_tag"].unique())
    configs = [c for c in CONFIG_ORDER if c in set(sub["config"])]

    # only plot metrics that actually made it through aggregation
    metrics = [m for m in METRICS if m[0] in sub.columns]
    if not metrics:
        print(f"skip {model}: no plottable metrics")
        continue

    # one row per metric — never hardcode the count, or zip() drops panels
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(14, 5 * len(metrics)),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    x     = np.arange(len(configs))
    width = 0.3 / max(len(gpus), 1)

    for ax, (col, ylabel, title) in zip(axes, metrics):
        err_col = col.replace("_mean", "_std")
        for i, gpu in enumerate(gpus):
            vals, errs = [], []
            for cfg in configs:
                row = sub[(sub["gpu_tag"] == gpu) & (sub["config"] == cfg)]
                vals.append(row[col].values[0] if len(row) else np.nan)
                if err_col in sub.columns and len(row):
                    e = row[err_col].values[0]
                    errs.append(0 if pd.isna(e) else e)
                else:
                    errs.append(0)
            offset = (i - (len(gpus) - 1) / 2) * width
            ax.bar(x + offset, vals, width, yerr=errs, label=gpu, capsize=3)

        # idle baseline is a useful reference on the power panel
        if col == "mean_power_w_mean" and "idle_power_w_mean" in sub.columns:
            idle = sub["idle_power_w_mean"].mean()
            if not pd.isna(idle):
                ax.axhline(idle, color="gray", ls=":", lw=1.2,
                           label=f"idle ≈ {idle:.0f} W")
                ax.legend(fontsize=12)

        ax.set_xticks(x + width * (len(gpus) - 1) / 2)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=14)
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    out = os.path.join(OUTDIR, f"hardware_{model.lower()}_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot     → {out}")
