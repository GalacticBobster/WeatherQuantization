"""
Combine ACC and RMSE plots from all configs into one figure each.
Auto-discovers configs across GPU dirs (outputs_V100, outputs_A100, ...).

Usage:  python combine_acc.py {dlwp|fcn}
"""
import matplotlib
matplotlib.use("Agg")
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

MODEL    = sys.argv[1].lower() if len(sys.argv) > 1 else "dlwp"
TRUTH    = "ERA5" if MODEL == "dlwp" else "GFS"
BASE     = "/glade/derecho/scratch/ananyo/WeatherQuantization"
OUTDIR   = os.path.join(BASE, "outputs_combined", f"{MODEL}_combined")
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = ["FP32","W8A8","W8A32","W4A32","W2A32","INT8_SMOOTHQUANT","INT4_AWQ"]
COLORS = {
    "FP32": "black",
    "W8A8": "royalblue",
    "W8A32": "limegreen",
    "W4A32": "gold",
    "W2A32": "crimson",
    "INT8_SMOOTHQUANT": "darkviolet",
    "INT4_AWQ": "steelblue",
}
LS = {"FP32": "--", **{c: "-" for c in CONFIGS if c != "FP32"}}

# ── auto-discover: look in all outputs_* dirs for each config ─
data = {}
for cfg in CONFIGS:
    npys = sorted(glob.glob(
        os.path.join(BASE, "outputs_A100", f"{MODEL}_bench", cfg, f"{MODEL}_{cfg}_acc.npy")))
    if not npys:
        print(f"  MISSING {cfg}")
        continue
    # take the largest (most complete) npy from any GPU dir
    npy = max(npys, key=lambda p: os.path.getsize(p))
    try:
        d = np.load(npy, allow_pickle=True).item()
        n_ae = len(d.get("ae", {}).get(d.get("vars", ["z500"])[0], []))
        n_re = len(d.get("re", {}).get(d.get("vars", ["z500"])[0], []))
        data[cfg] = d
        print(f"  Loaded {cfg} from {os.path.basename(os.path.dirname(os.path.dirname(npy)))}  (n_re={n_re}, n_ae={n_ae})")
    except Exception as e:
        print(f"  FAILED {cfg}: {e}")

if not data:
    sys.exit("No npy files found.")

# ── layout ────────────────────────────────────────────────────
ref         = data.get("FP32") or next(iter(data.values()))
MODEL_VARS  = ref["vars"]
leads       = np.array(ref["leads"])
ncols       = min(4, len(MODEL_VARS))
nrows       = int(np.ceil(len(MODEL_VARS)/ncols))

def render(source_key, ylabel, tag, hline=None, ylim=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    n_plot = 0
    for i, (ax, v) in enumerate(zip(axes, MODEL_VARS)):
        any_line = False
        for cfg in CONFIGS:
            if cfg not in data: continue
            vals = data[cfg].get(source_key, {}).get(v, [])
            if not vals: continue
            arr  = np.array(vals)
            m, s = arr.mean(0), arr.std(0)
            ax.plot(leads, m, color=COLORS[cfg], lw=1.8, ls=LS[cfg], label=cfg)
            ax.fill_between(leads, m-s, m+s, color=COLORS[cfg], alpha=0.10)
            any_line = True
        if not any_line: ax.set_visible(False); continue
        if hline is not None: ax.axhline(hline, color="gray", lw=0.8, ls=":", alpha=0.6)
        if ylim is not None:  ax.set_ylim(ylim)
        ax.set_title(v, fontweight="bold", fontsize=15); ax.set_xlabel("Lead time (h)", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15); ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i == 0: ax.legend(fontsize=12, loc="best")   # legend on first panel only
        n_plot += 1
    for ax in axes[len(MODEL_VARS):]: ax.set_visible(False)
    if n_plot == 0: plt.close(fig); print(f"  No data for {tag}"); return
    #fig.suptitle(f"{MODEL.upper()} {ylabel} — all configs",
    #             fontweight="bold", fontsize=14)
    out = os.path.join(OUTDIR, f"{MODEL}_combined_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")

render("re", f"RMSE ({TRUTH})",  "rmse_vs_truth")
render("rf", "RMSE (FP 32)",       "rmse_vs_fp32")
render("ae", "ACC",                "acc", hline=0.6, ylim=(-0.1, 1.05))
