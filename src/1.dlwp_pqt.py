import os
import numpy as np
import matplotlib.pyplot as plt
import zarr, torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]
EVAL_DATES  = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]
NSTEPS      = 10
CALIB_DATE  = "2020-01-01"
CALIB_STEPS = 8
OUTDIR      = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs/combined"

def make_cfg(wb=8, ab=None):
    return {"quant_cfg": [
        {"quantizer_name": "*",                "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": wb, "axis": 0}},
        *([] if ab is None else [{"quantizer_name": "*input_quantizer",
                                   "cfg": {"num_bits": ab, "axis": None}}]),
        {"quantizer_name": "equatorial_downsample.0.*", "enable": False},
        {"quantizer_name": "polar_downsample.0.*",      "enable": False},
        {"quantizer_name": "equatorial_last.*",         "enable": False},
        {"quantizer_name": "polar_last.*",              "enable": False},
    ], "algorithm": "max"}

EXPERIMENTS = {
    "FP32":             (None,                      "black",         "-"),
    "W8A8":             (make_cfg(8, 8),             "steelblue",    "--"),
    "W8A32":            (make_cfg(8, None),          "mediumseagreen","--"),
    "W4A32":            (make_cfg(4, None),          "darkorange",   "--"),
    "INT8_SMOOTHQUANT": (mtq.INT8_SMOOTHQUANT_CFG,   "firebrick",    "--"),
    "INT4_AWQ":         (mtq.INT4_AWQ_CFG,           "purple",       "--"),
}

os.makedirs(OUTDIR, exist_ok=True)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5    = ARCO()
package = DLWP.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

def run_fcast(model, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(date_str)], NSTEPS, model, era5, io)
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in VARIABLES if v in avail(io)}, leads

def get_truth(date_str, probe):
    init, out = datetime.fromisoformat(date_str), {v: [] for v in VARIABLES}
    for s in range(NSTEPS + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=s*6)], 0, probe, era5, io)
        for v in VARIABLES:
            if v in avail(io): out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in VARIABLES if out[v]}

probe  = DLWP.load_model(package).to(device).eval()
_io    = ZarrBackend()
with torch.no_grad(): _io = run.deterministic([datetime.fromisoformat(CALIB_DATE)], 1, probe, era5, _io)
w      = np.cos(np.deg2rad(_io["lat"][:])); w = (w/w.mean())[np.newaxis,:,np.newaxis]
lrmse  = lambda a, b: np.sqrt(np.mean(w*(a-b)**2, axis=(-1,-2)))
agg    = lambda lst: (np.stack(lst).mean(0), np.stack(lst).std(0))

results, fp32_fores, leads = {}, {}, None
for exp, (cfg, col, ls) in EXPERIMENTS.items():
    print(f"\n{exp}")
    m = DLWP.load_model(package).to(device).eval()
    if cfg is not None:
        ci = datetime.fromisoformat(CALIB_DATE)
        def fwd(inner, _m=m):
            _m.model = inner
            with torch.no_grad(): run.deterministic([ci], CALIB_STEPS, _m, era5, ZarrBackend())
        m.model = mtq.quantize(m.model, cfg, fwd)
    re, rf = {v: [] for v in VARIABLES}, {v: [] for v in VARIABLES}
    for d in EVAL_DATES:
        truth      = get_truth(d, probe)
        fore, leads = run_fcast(m, d)
        if exp == "FP32": fp32_fores[d] = fore
        for v in VARIABLES:
            if v not in truth: continue
            re[v].append(lrmse(fore[v], truth[v]))
            if exp != "FP32" and v in fp32_fores.get(d, {}):
                rf[v].append(lrmse(fore[v], fp32_fores[d][v]))
    results[exp] = {"era5": {v: re[v] for v in VARIABLES if re[v]},
                    "fp32": {v: rf[v] for v in VARIABLES if rf[v]}}
    del m

ncols = min(4, len(VARIABLES)); nrows = int(np.ceil(len(VARIABLES)/ncols))

def make_fig(title, fname):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax in axes[len(VARIABLES):]: ax.set_visible(False)
    fig.suptitle(title, fontweight="bold")
    return fig, axes

# Plot 1: vs ERA5
fig1, ax1 = make_fig(f"DLWP RMSE vs ERA5 — all configs | {len(EVAL_DATES)} dates | mean ±1σ",
                     "combined_rmse_vs_era5.png")
for ax, v in zip(ax1, VARIABLES):
    for exp, (_, col, ls) in EXPERIMENTS.items():
        d = results[exp]["era5"]
        if v not in d: continue
        m, s = agg(d[v])
        ax.plot(leads, m, color=col, lw=2, ls=ls, label=exp)
        ax.fill_between(leads, m-s, m+s, color=col, alpha=0.10)
    ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE vs ERA5"); ax.grid(True, alpha=0.3)
ax1[0].legend(fontsize=7, ncol=2)
fig1.savefig(os.path.join(OUTDIR, "combined_rmse_vs_era5.png"), dpi=150)

# Plot 2: vs FP32
fig2, ax2 = make_fig(f"DLWP RMSE vs FP32 — all configs | {len(EVAL_DATES)} dates | mean ±1σ",
                     "combined_rmse_vs_fp32.png")
for ax, v in zip(ax2, VARIABLES):
    for exp, (_, col, ls) in EXPERIMENTS.items():
        if exp == "FP32": continue
        d = results[exp]["fp32"]
        if v not in d: continue
        m, s = agg(d[v])
        ax.plot(leads, m, color=col, lw=2, label=exp)
        ax.fill_between(leads, m-s, m+s, color=col, alpha=0.10)
    ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE vs FP32"); ax.grid(True, alpha=0.3)
ax2[0].legend(fontsize=7, ncol=2)
fig2.savefig(os.path.join(OUTDIR, "combined_rmse_vs_fp32.png"), dpi=150)

# Plot 3: FP32 self-consistency
fp32_a = DLWP.load_model(package).to(device).eval()
fp32_b = DLWP.load_model(package).to(device).eval()
self_rmse = {v: [] for v in VARIABLES}
for d in EVAL_DATES:
    fa, _ = run_fcast(fp32_a, d); fb, _ = run_fcast(fp32_b, d)
    for v in VARIABLES:
        if v in fa and v in fb: self_rmse[v].append(lrmse(fa[v], fb[v]))

fig3, ax3 = make_fig(f"FP32 self-consistency — numerical noise floor | {len(EVAL_DATES)} dates",
                     "fp32_self_consistency.png")
for ax, v in zip(ax3, VARIABLES):
    if not self_rmse[v]: ax.set_visible(False); continue
    m, s = agg(self_rmse[v])
    ax.plot(leads, m, color="black", lw=2)
    ax.fill_between(leads, m-s, m+s, color="black", alpha=0.15)
    ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE FP32 vs FP32"); ax.grid(True, alpha=0.3)
fig3.savefig(os.path.join(OUTDIR, "fp32_self_consistency.png"), dpi=150)

plt.close("all")
print("\nAll done.")
