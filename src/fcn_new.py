"""
FCN PTQ benchmark: skill (RMSE + ACC) and hardware metrics across configs.
Uses ARCO ERA5 (via ArcoFCN wrapper) for initial conditions AND truth.
WB2 ERA5 climatology for ACC anomaly reference.
Analytical FLOPs from architecture specs.
"""
import matplotlib
matplotlib.use("Agg")
import argparse, os, csv, time, platform
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
from earth2studio.data import WB2Climatology
from earth2studio.models.px.fcn import FCN
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

from arco_for_fcn import ArcoFCN   # our ARCO ERA5 → FCN 26-channel wrapper

# ═══════════════════════════════════════════════════════════════
# in the VARIABLES list — remove "u10", "v10" since output is now called u10m
VARIABLES = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv",
             "u10m", "v10m",         # matches FCN output names
             "u500", "v500", "u850", "v850"]
EVAL_DATES  = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]
NSTEPS      = 10
CALIB_STEPS = 8
MODEL_NAME  = "FCN"

FLOPS = {"params_M": 74.0, "flops_step_G": 2400.0, "flops_forecast_G": 24000.0}
# ═══════════════════════════════════════════════════════════════

def make_cfg(wb=8, ab=None):
    quant_cfg = {
        "*":                        {"enable": False},
        "*weight_quantizer":         {"num_bits": wb, "axis": 0},
        "*norm*":                    {"enable": False},
        "*pos_drop*":                {"enable": False},
        "model.patch_embed.*":       {"enable": False},
        "model.head.*":              {"enable": False},
    }
    if ab is not None:
        quant_cfg["*input_quantizer"] = {"num_bits": ab, "axis": None}
    return {"quant_cfg": quant_cfg, "algorithm": "max"}

CONFIGS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,     # may fail on E2S; see limitations
}
BITS = {"FP32":32,"W8A8":8,"W8A32":8,"W4A32":4,"W2A32":2,
        "INT8_SMOOTHQUANT":8,"INT4_AWQ":4}

# ── args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(CONFIGS))
parser.add_argument("--outdir", required=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

slug    = f"fcn_{args.config}"
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5    = ArcoFCN()                # replaces the old GFS() data source
clim    = WB2Climatology()
package = FCN.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

# ── helpers ───────────────────────────────────────────────────
def gpu_info():
    if not torch.cuda.is_available(): return {"gpu_name":"CPU","gpu_mem_total_mb":0}
    p = torch.cuda.get_device_properties(0)
    return {"gpu_name": p.name, "gpu_mem_total_mb": p.total_memory/1e6}

def gpu_power_w():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi","--query-gpu=power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return float(r.stdout.strip().split("\n")[0])
    except: return float("nan")

def model_size_mb(m, bits): return sum(p.numel() for p in m.parameters())*bits/8/1e6

def benchmark_time(m, n_runs=5):
    init = datetime.fromisoformat(EVAL_DATES[0])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        io = ZarrBackend()
        with torch.no_grad(): run.deterministic([init], NSTEPS, m, era5, io)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return 1000*(time.perf_counter()-t0)/n_runs

# ── probe ─────────────────────────────────────────────────────
_probe = FCN.load_model(package).to(device).eval()
_io    = ZarrBackend()
with torch.no_grad():
    _io = run.deterministic([datetime.fromisoformat(EVAL_DATES[0])], 1, _probe, era5, _io)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS = [v for v in VARIABLES if v in avail(_io) and v not in coord_keys]
w_lat      = np.cos(np.deg2rad(_io["lat"][:])); w_lat = (w_lat/w_lat.mean())[np.newaxis,:,np.newaxis]
print(f"Variables: {MODEL_VARS}")
del _probe

def lat_rmse(a, b): return np.sqrt(np.mean(w_lat*(a-b)**2, axis=(-1,-2)))

def lat_acc(f, t, c):
    fa, ta = f - c, t - c
    num    = np.sum(w_lat*fa*ta, axis=(-1,-2))
    denom  = np.sqrt(np.sum(w_lat*fa**2, axis=(-1,-2)) *
                     np.sum(w_lat*ta**2, axis=(-1,-2)))
    return np.where(denom > 0, num/denom, 0.0)

def get_climatology(date_str):
    init = datetime.fromisoformat(date_str)
    out  = {v: [] for v in MODEL_VARS}
    for s in range(NSTEPS + 1):
        vt = init + timedelta(hours=s*6)
        try:
            da = clim(time=vt, variable=MODEL_VARS)
            for v in MODEL_VARS:
                field = np.squeeze(da.sel(variable=v).values)
                # trim 721→720 lats if needed (climatology has south pole, FCN doesn't)
                if field.shape[0] == 721 and w_lat.shape[1] == 720:
                    field = field[:720]
                out[v].append(field)
        except Exception as e:
            print(f"  Clim fetch failed at {vt}: {e}")
            return None
    return {v: np.stack(out[v], 0) for v in MODEL_VARS}

# ── quantize ──────────────────────────────────────────────────
calib_init = datetime.fromisoformat(EVAL_DATES[0])
with torch.inference_mode(False):
    model = FCN.load_model(package).to(device).eval()
    for p in model.model.parameters(): p.data = p.data.clone()
    for b in model.model.buffers():    b.data = b.data.clone()

    if CONFIGS[args.config] is not None:
        def fwd(inner):
            model.model = inner
            with torch.inference_mode(False):
                run.deterministic([calib_init], CALIB_STEPS, model, era5, ZarrBackend())
        print(f"Calibrating {args.config} …")
        model.model = mtq.quantize(model.model, CONFIGS[args.config], fwd)

fp32_model  = FCN.load_model(package).to(device).eval()
probe_model = FCN.load_model(package).to(device).eval()

# ── hardware ──────────────────────────────────────────────────
hw       = gpu_info()
theo_mb  = model_size_mb(model.model, BITS[args.config])
fp32_mb  = model_size_mb(model.model, 32)
power1   = gpu_power_w()
print("Benchmarking inference time …")
infer_ms = benchmark_time(model)
power2   = gpu_power_w()
avg_power= np.nanmean([power1, power2])

# ── hardware CSV early ────────────────────────────────────────
import modelopt
csv_path = os.path.join(args.outdir, "hardware_metrics.csv")
new      = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "model","config","gpu_name","gpu_mem_total_mb",
        "theoretical_size_mb","fp32_size_mb","compression_ratio",
        "inference_ms_per_forecast","avg_power_w",
        "params_M","flops_step_G","flops_forecast_G",
        "torch_version","modelopt_version","platform"])
    if new: w.writeheader()
    w.writerow({
        "model":                    MODEL_NAME,
        "config":                   args.config,
        "gpu_name":                 hw["gpu_name"],
        "gpu_mem_total_mb":         f"{hw['gpu_mem_total_mb']:.0f}",
        "theoretical_size_mb":      f"{theo_mb:.1f}",
        "fp32_size_mb":             f"{fp32_mb:.1f}",
        "compression_ratio":        f"{fp32_mb/theo_mb:.1f}x",
        "inference_ms_per_forecast":f"{infer_ms:.0f}",
        "avg_power_w":              f"{avg_power:.1f}",
        "params_M":                 f"{FLOPS['params_M']:.2f}",
        "flops_step_G":             f"{FLOPS['flops_step_G']:.2f}",
        "flops_forecast_G":         f"{FLOPS['flops_forecast_G']:.2f}",
        "torch_version":            torch.__version__,
        "modelopt_version":         modelopt.__version__,
        "platform":                 platform.node(),
    })
print(f"Hardware CSV → {csv_path}")

# ── skill evaluation ──────────────────────────────────────────
def run_fcast(m, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(date_str)], NSTEPS, m, era5, io)
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in MODEL_VARS if v in avail(io)}, leads

def get_truth(date_str):
    init, out = datetime.fromisoformat(date_str), {v: [] for v in MODEL_VARS}
    for s in range(NSTEPS + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=s*6)], 0, probe_model, era5, io)
        for v in MODEL_VARS:
            if v in avail(io): out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in MODEL_VARS if out[v]}

re, rf, r32 = {v:[] for v in MODEL_VARS}, {v:[] for v in MODEL_VARS}, {v:[] for v in MODEL_VARS}
ae, af      = {v:[] for v in MODEL_VARS}, {v:[] for v in MODEL_VARS}
leads = None

for d in EVAL_DATES:
    print(f"  Eval {d}")
    truth        = get_truth(d)
    q_fore, leads= run_fcast(model,      d)
    f_fore, _    = run_fcast(fp32_model, d)
    clim_d       = get_climatology(d)
    for v in MODEL_VARS:
        if v not in truth: continue
        re[v].append(lat_rmse(q_fore[v], truth[v]))
        r32[v].append(lat_rmse(f_fore[v], truth[v]))
        rf[v].append(lat_rmse(q_fore[v], f_fore[v]))
        if clim_d and v in clim_d:
            ae[v].append(lat_acc(q_fore[v], truth[v], clim_d[v]))
            af[v].append(lat_acc(f_fore[v], truth[v], clim_d[v]))

# ── NPY save early ────────────────────────────────────────────
def safe(d): return {v: [a.tolist() for a in d[v]] for v in MODEL_VARS}
payload = {
    "vars":  MODEL_VARS,
    "leads": leads.tolist() if leads is not None else [],
    "ae": safe(ae), "af": safe(af),
    "re": safe(re), "rf": safe(rf), "r32": safe(r32),
}
np_path = os.path.join(args.outdir, f"{slug}_acc.npy")
try:
    with open(np_path, 'wb') as f:
        np.save(f, payload, allow_pickle=True)
        f.flush(); os.fsync(f.fileno())
    print(f"NPY → {np_path}")
except Exception as e:
    print(f"NPY save error: {e}")

# ── plots ─────────────────────────────────────────────────────
def safe_agg(lst):
    if not lst: return None, None
    arr = np.stack(lst); return arr.mean(0), arr.std(0)

ncols = min(4, len(MODEL_VARS))
nrows = int(np.ceil(len(MODEL_VARS)/ncols))

def plot_panel(data_q, data_ref, ylabel, tag, hline=None, ylim=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax, v in zip(axes, MODEL_VARS):
        m, s = safe_agg(data_q[v])
        if m is None: ax.set_visible(False); continue
        ax.plot(leads, m, color="firebrick", lw=2, label=args.config)
        ax.fill_between(leads, m-s, m+s, color="firebrick", alpha=0.15)
        if data_ref is not None:
            m2, s2 = safe_agg(data_ref[v])
            if m2 is not None:
                ax.plot(leads, m2, color="steelblue", lw=2, ls="--", label="FP32")
                ax.fill_between(leads, m2-s2, m2+s2, color="steelblue", alpha=0.10)
        if hline is not None: ax.axhline(hline, color="gray", lw=0.8, ls=":", alpha=0.6)
        if ylim is not None:  ax.set_ylim(ylim)
        ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes[len(MODEL_VARS):]: ax.set_visible(False)
    fig.suptitle(f"{MODEL_NAME} {ylabel} — {args.config} | {len(EVAL_DATES)} dates",
                 fontweight="bold")
    fig.savefig(os.path.join(args.outdir, f"{slug}_{tag}.png"), dpi=150); plt.close(fig)
    print(f"Saved {tag}.png")

try: plot_panel(re, r32, "Lat-wtd RMSE vs ERA5", "rmse_vs_era5")
except Exception as e: print(f"RMSE plot failed: {e}")
try: plot_panel(rf, None, "Lat-wtd RMSE vs FP32", "rmse_vs_fp32")
except Exception as e: print(f"RMSE-FP32 plot failed: {e}")
try: plot_panel(ae, af,  "Lat-wtd ACC", "acc", hline=0.6, ylim=(-0.1, 1.05))
except Exception as e: print(f"ACC plot failed: {e}")

print(f"\nDone: {slug}")
print(f"  Size:  {theo_mb:.1f} MB ({fp32_mb/theo_mb:.1f}x)")
print(f"  Time:  {infer_ms:.0f} ms/forecast")
print(f"  Power: {avg_power:.1f} W")
print(f"  FLOPs: {FLOPS['flops_forecast_G']:.1f} G/forecast")
