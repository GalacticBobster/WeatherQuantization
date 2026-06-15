import matplotlib
matplotlib.use("Agg")
import argparse, os, csv, time, platform
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
from earth2studio.data import GFS
from earth2studio.models.px import FCN
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]
EVAL_DATES  = ["2022-01-01","2022-04-01","2022-07-01","2022-10-01"]
NSTEPS      = 20
CALIB_STEPS = 8

def make_cfg(wb=8, ab=None):
    return {"quant_cfg": [
        {"quantizer_name": "*",                  "enable": False},
        {"quantizer_name": "*weight_quantizer",   "cfg": {"num_bits": wb, "axis": 0}},
        *([] if ab is None else [
          {"quantizer_name": "*input_quantizer",  "cfg": {"num_bits": ab, "axis": None}}]),
        {"quantizer_name": "*norm*",              "enable": False},
        {"quantizer_name": "*pos_drop*",          "enable": False},
        {"quantizer_name": "model.patch_embed.*", "enable": False},
        {"quantizer_name": "model.head.*",        "enable": False},
    ], "algorithm": "max"}

CONFIGS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

BITS = {"FP32":32,"W8A8":8,"W8A32":8,"W4A32":4,"W2A32":2,"INT8_SMOOTHQUANT":8,"INT4_AWQ":4}

parser = argparse.ArgumentParser()
parser.add_argument("--config",  required=True, choices=list(CONFIGS))
parser.add_argument("--outdir",  required=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gfs     = GFS()
package = FCN.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

def gpu_info():
    if not torch.cuda.is_available():
        return {"gpu_name": "CPU", "gpu_mem_total_mb": 0}
    props = torch.cuda.get_device_properties(0)
    return {"gpu_name": props.name, "gpu_mem_total_mb": props.total_memory / 1e6}

def gpu_power_w():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        return float(r.stdout.strip().split("\n")[0])
    except: return float("nan")

def model_size_mb(m, bits):
    return sum(p.numel() for p in m.parameters()) * bits / 8 / 1e6

def benchmark_time(m, n_runs=5):
    init = datetime.fromisoformat(EVAL_DATES[0])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        io = ZarrBackend()
        with torch.no_grad():
            run.deterministic([init], NSTEPS, m, gfs, io)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n_runs

# probe
_probe = FCN.load_model(package).to(device).eval()
_io    = ZarrBackend()
with torch.no_grad():
    _io = run.deterministic([datetime.fromisoformat(EVAL_DATES[0])], 1, _probe, gfs, _io)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS = [v for v in VARIABLES if v in avail(_io) and v not in coord_keys]
w_lat      = np.cos(np.deg2rad(_io["lat"][:])); w_lat = (w_lat/w_lat.mean())[np.newaxis,:,np.newaxis]
lat_rmse   = lambda a, b: np.sqrt(np.mean(w_lat*(a-b)**2, axis=(-1,-2)))
del _probe

# load and quantize
calib_init = datetime.fromisoformat(EVAL_DATES[0])
model      = FCN.load_model(package).to(device).eval()
if CONFIGS[args.config] is not None:
    def fwd(inner):
        model.model = inner
        with torch.no_grad():
            run.deterministic([calib_init], CALIB_STEPS, model, gfs, ZarrBackend())
    print(f"Calibrating {args.config} …")
    model.model = mtq.quantize(model.model, CONFIGS[args.config], fwd)

fp32_model  = FCN.load_model(package).to(device).eval()
probe_model = FCN.load_model(package).to(device).eval()

# hardware metrics
hw       = gpu_info()
theo_mb  = model_size_mb(model.model, BITS[args.config])
fp32_mb  = model_size_mb(model.model, 32)
power_w  = gpu_power_w()
print("Benchmarking inference time …")
infer_ms = benchmark_time(model)
power_w2 = gpu_power_w()
avg_power = np.nanmean([power_w, power_w2])

# skill evaluation
def run_fcast(m, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(date_str)], NSTEPS, m, gfs, io)
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in MODEL_VARS if v in avail(io)}, leads

def get_truth(date_str):
    init, out = datetime.fromisoformat(date_str), {v: [] for v in MODEL_VARS}
    for s in range(NSTEPS + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=s*6)], 0, probe_model, gfs, io)
        for v in MODEL_VARS:
            if v in avail(io): out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in MODEL_VARS if out[v]}

re, rf, r32 = {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}
leads = None
for d in EVAL_DATES:
    print(f"  {d}")
    truth        = get_truth(d)
    q_fore, leads= run_fcast(model,      d)
    f_fore, _    = run_fcast(fp32_model, d)
    for v in MODEL_VARS:
        if v not in truth: continue
        re[v].append(lat_rmse(q_fore[v], truth[v]))
        r32[v].append(lat_rmse(f_fore[v], truth[v]))
        rf[v].append(lat_rmse(q_fore[v], f_fore[v]))

agg   = lambda lst: (np.stack(lst).mean(0), np.stack(lst).std(0))
slug  = f"fcn_{args.config}"
ncols = min(4, len(MODEL_VARS))
nrows = int(np.ceil(len(MODEL_VARS) / ncols))

for metric, data_q, data_ref, ylabel, tag in [
    ("era5", re,  r32, "Lat-wtd RMSE vs GFS",  "rmse_vs_gfs"),
    ("fp32", rf,  None,"Lat-wtd RMSE vs FP32",  "rmse_vs_fp32"),
]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax, v in zip(axes, MODEL_VARS):
        if not data_q[v]: ax.set_visible(False); continue
        m, s = agg(data_q[v])
        ax.plot(leads, m, color="firebrick", lw=2, label=args.config)
        ax.fill_between(leads, m-s, m+s, color="firebrick", alpha=0.15)
        if data_ref and data_ref[v]:
            m2, s2 = agg(data_ref[v])
            ax.plot(leads, m2, color="steelblue", lw=2, ls="--", label="FP32")
            ax.fill_between(leads, m2-s2, m2+s2, color="steelblue", alpha=0.10)
        ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes[len(MODEL_VARS):]: ax.set_visible(False)
    fig.suptitle(f"FCN {ylabel} — {args.config} | {len(EVAL_DATES)} dates | mean ±1σ",
                 fontweight="bold")
    fig.savefig(os.path.join(args.outdir, f"{slug}_{tag}.png"), dpi=150)
    plt.close(fig)

# hardware CSV
csv_path = os.path.join(args.outdir, "hardware_metrics.csv")
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "model","config","gpu_name","gpu_mem_total_mb",
        "theoretical_size_mb","fp32_size_mb","compression_ratio",
        "inference_ms_per_forecast","avg_power_w","platform"])
    if write_header: w.writeheader()
    w.writerow({
        "model":                    "FCN",
        "config":                   args.config,
        "gpu_name":                 hw["gpu_name"],
        "gpu_mem_total_mb":         hw["gpu_mem_total_mb"],
        "theoretical_size_mb":      f"{theo_mb:.1f}",
        "fp32_size_mb":             f"{fp32_mb:.1f}",
        "compression_ratio":        f"{fp32_mb/theo_mb:.1f}x",
        "inference_ms_per_forecast":f"{infer_ms:.0f}",
        "avg_power_w":              f"{avg_power:.1f}",
        "platform":                 platform.node(),
    })
print(f"\nHardware metrics: {csv_path}")
print(f"  Size: {theo_mb:.1f} MB ({fp32_mb/theo_mb:.1f}x compression)")
print(f"  Inference: {infer_ms:.0f} ms/forecast")
print(f"  Power: {avg_power:.1f} W")
# ── hardware metrics plot ─────────────────────────────────────
import pandas as pd

# read all configs' CSVs from sibling directories
records = []
base = os.path.dirname(args.outdir)
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d, "hardware_metrics.csv")
    if os.path.exists(p):
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

if len(records) > 1:
    configs  = [r["config"]                        for r in records]
    sizes    = [float(r["theoretical_size_mb"])     for r in records]
    compress = [float(r["compression_ratio"].replace("x","")) for r in records]
    times    = [float(r["inference_ms_per_forecast"])for r in records]
    powers   = [float(r["avg_power_w"]) if r["avg_power_w"] != "nan" else 0
                for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(configs)))

    # size
    axes[0].barh(configs, sizes, color=colors)
    axes[0].set_xlabel("Model size (MB)")
    axes[0].set_title("Theoretical Memory", fontweight="bold")
    axes[0].axvline(sizes[0], color="black", lw=1, ls="--", alpha=0.5)

    # inference time
    axes[1].barh(configs, times, color=colors)
    axes[1].set_xlabel("ms per forecast")
    axes[1].set_title("Inference Time", fontweight="bold")
    axes[1].axvline(times[0], color="black", lw=1, ls="--", alpha=0.5)

    # power
    if any(p > 0 for p in powers):
        axes[2].barh(configs, powers, color=colors)
        axes[2].set_xlabel("Watts")
        axes[2].set_title("GPU Power Draw", fontweight="bold")
        axes[2].axvline(powers[0], color="black", lw=1, ls="--", alpha=0.5)
    else:
        axes[2].text(0.5, 0.5, "Power data\nnot available",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("GPU Power Draw", fontweight="bold")

    model_name = records[0]["model"]
    gpu_name   = records[0]["gpu_name"]
    fig.suptitle(f"{model_name} hardware metrics — {gpu_name}", fontweight="bold")
    fig.savefig(os.path.join(base, f"{model_name.lower()}_hardware_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"Hardware plot saved: {model_name.lower()}_hardware_metrics.png")

print("Done.")
