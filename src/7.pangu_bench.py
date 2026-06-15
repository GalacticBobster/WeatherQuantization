import matplotlib
matplotlib.use("Agg")
import argparse, os, csv, time, platform
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
import onnxruntime as ort
import modelopt.onnx.quantization as mto
from earth2studio.data import ARCO
from earth2studio.models.px.pangu import Pangu6
from earth2studio.io import ZarrBackend
import earth2studio.run as run

# ── Pangu ONNX paths ──────────────────────────────────────────
ONNX_6H  = "/glade/derecho/scratch/ananyo/e2s_cache/pangu/pangu_weather_6.onnx"
ONNX_24H = "/glade/derecho/scratch/ananyo/e2s_cache/pangu/pangu_weather_24.onnx"

VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]
EVAL_DATES  = ["2020-01-01","2020-04-01","2020-07-01","2020-10-01"]
NSTEPS      = 4    # Pangu6 runs in 6h steps
CALIB_STEPS = 4
CALIB_DATE  = "2020-01-01"

# quantization configs for ONNX
CONFIGS = {
    "FP32":  None,
    "INT8":  "int8",
    "FP8":   "fp8",
}

parser = argparse.ArgumentParser()
parser.add_argument("--config",  required=True, choices=list(CONFIGS))
parser.add_argument("--outdir",  required=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

era5  = ARCO()
avail = lambda io: list(zarr.open(io.store, mode="r").keys())

# ── GPU info ──────────────────────────────────────────────────
def gpu_info():
    if not torch.cuda.is_available():
        return {"gpu_name": "CPU", "gpu_mem_total_mb": 0}
    props = torch.cuda.get_device_properties(0)
    return {"gpu_name": props.name, "gpu_mem_total_mb": props.total_memory/1e6}

def gpu_power_w():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        return float(r.stdout.strip().split("\n")[0])
    except: return float("nan")

def onnx_size_mb(path):
    return os.path.getsize(path) / 1e6

# ── calibration data generator ────────────────────────────────
def make_calib_data(n_samples=4):
    return {
        "input":         np.stack([
                             np.random.randn(5, 13, 721, 1440).astype(np.float32)
                             for _ in range(n_samples)], axis=0),
        "input_surface": np.stack([
                             np.random.randn(4, 721, 1440).astype(np.float32)
                             for _ in range(n_samples)], axis=0),
    }

# ── quantize ONNX ─────────────────────────────────────────────
onnx_6h_out  = os.path.join(args.outdir, f"pangu_6h_{args.config}.onnx")
onnx_24h_out = os.path.join(args.outdir, f"pangu_24h_{args.config}.onnx")

if args.config != "FP32":
    print(f"Quantizing Pangu ONNX to {args.config} …")
    print(f"Calib samples: {CALIB_STEPS}")
    calib_data = make_calib_data(max(1, CALIB_STEPS))
    #calib_data = make_calib_data(CALIB_STEPS)

    mto.quantize(
        onnx_path        = ONNX_6H,
        quantize_mode    = CONFIGS[args.config],
        output_path      = onnx_6h_out,
        calibration_data = calib_data,
    )
    mto.quantize(
        onnx_path        = ONNX_24H,
        quantize_mode    = CONFIGS[args.config],
        output_path      = onnx_24h_out,
        calibration_data = calib_data,
    )
    print("Quantization done.")
else:
    onnx_6h_out  = ONNX_6H
    onnx_24h_out = ONNX_24H

# ── load quantized model into E2S ────────────────────────────
providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
model = Pangu6.load_model(Pangu6.load_default_package()).eval()
#model.ort   = ort.InferenceSession(onnx_6h_out,  providers=providers)
#model.ort24 = ort.InferenceSession(onnx_24h_out, providers=providers)

model.ort   = onnx_6h_out
model.ort24 = onnx_24h_out
print(f"Loaded {args.config} Pangu6 model")

fp32_model = Pangu6.load_model(Pangu6.load_default_package()).eval()

# ── hardware metrics ──────────────────────────────────────────
hw        = gpu_info()
fp32_mb   = onnx_size_mb(ONNX_6H)
quant_mb  = onnx_size_mb(onnx_6h_out)
power_pre = gpu_power_w()

def benchmark_time(m, n_runs=3):
    init = datetime.fromisoformat(EVAL_DATES[0])
    t0   = time.perf_counter()
    for _ in range(n_runs):
        io = ZarrBackend()
        with torch.no_grad():
            run.deterministic([init], NSTEPS, m, era5, io)
    return 1000 * (time.perf_counter() - t0) / n_runs

print("Benchmarking inference time …")
infer_ms  = benchmark_time(model)
power_post= gpu_power_w()
avg_power = np.nanmean([power_pre, power_post])

# ── lat-weighted RMSE ─────────────────────────────────────────
_io = ZarrBackend()
with torch.no_grad():
    run.deterministic([datetime.fromisoformat(CALIB_DATE)], 1, model, era5, _io)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS = [v for v in VARIABLES if v in avail(_io) and v not in coord_keys]
w_lat      = np.cos(np.deg2rad(_io["lat"][:])); w_lat = (w_lat/w_lat.mean())[np.newaxis,:,np.newaxis]
lat_rmse   = lambda a, b: np.sqrt(np.mean(w_lat*(a-b)**2, axis=(-1,-2)))
print(f"Variables: {MODEL_VARS}")

def run_fcast(m, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(date_str)], NSTEPS, m, era5, io)
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in MODEL_VARS if v in avail(io)}, leads

def get_truth(date_str, probe):
    init, out = datetime.fromisoformat(date_str), {v: [] for v in MODEL_VARS}
    for s in range(NSTEPS + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=s*6)], 0, probe, era5, io)
        for v in MODEL_VARS:
            if v in avail(io): out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in MODEL_VARS if out[v]}

probe_model = Pangu6.load_model(Pangu6.load_default_package()).eval()
re, rf, r32 = {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}
leads = None

for d in EVAL_DATES:
    print(f"  {d}")
    truth        = get_truth(d, probe_model)
    q_fore, leads= run_fcast(model,      d)
    f_fore, _    = run_fcast(fp32_model, d)
    for v in MODEL_VARS:
        if v not in truth: continue
        re[v].append(lat_rmse(q_fore[v], truth[v]))
        r32[v].append(lat_rmse(f_fore[v], truth[v]))
        rf[v].append(lat_rmse(q_fore[v], f_fore[v]))

agg   = lambda lst: (np.stack(lst).mean(0), np.stack(lst).std(0))
slug  = f"pangu_{args.config}"
ncols = min(4, len(MODEL_VARS))
nrows = int(np.ceil(len(MODEL_VARS) / ncols))

# ── skill plots ───────────────────────────────────────────────
for data_q, data_ref, ylabel, tag in [
    (re,  r32, "Lat-wtd RMSE vs ERA5", "rmse_vs_era5"),
    (rf,  None,"Lat-wtd RMSE vs FP32", "rmse_vs_fp32"),
]:
    if not any(data_q[v] for v in MODEL_VARS): continue
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
    fig.suptitle(f"Pangu6 {ylabel} — {args.config} | {len(EVAL_DATES)} dates | mean ±1σ",
                 fontweight="bold")
    fig.savefig(os.path.join(args.outdir, f"{slug}_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {slug}_{tag}.png")

# ── hardware CSV ──────────────────────────────────────────────
import modelopt
csv_path     = os.path.join(args.outdir, "hardware_metrics.csv")
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "model","config","gpu_name","gpu_mem_total_mb",
        "onnx_size_mb","fp32_onnx_size_mb","compression_ratio",
        "inference_ms_per_forecast","avg_power_w",
        "torch_version","modelopt_version","platform"])
    if write_header: w.writeheader()
    w.writerow({
        "model":                    "Pangu6",
        "config":                   args.config,
        "gpu_name":                 hw["gpu_name"],
        "gpu_mem_total_mb":         f"{hw['gpu_mem_total_mb']:.0f}",
        "onnx_size_mb":             f"{quant_mb:.1f}",
        "fp32_onnx_size_mb":        f"{fp32_mb:.1f}",
        "compression_ratio":        f"{fp32_mb/quant_mb:.1f}x" if quant_mb > 0 else "1.0x",
        "inference_ms_per_forecast":f"{infer_ms:.0f}",
        "avg_power_w":              f"{avg_power:.1f}",
        "torch_version":            torch.__version__,
        "modelopt_version":         modelopt.__version__,
        "platform":                 platform.node(),
    })

print(f"\nHardware metrics: {csv_path}")
print(f"  ONNX size: {quant_mb:.1f} MB  (FP32: {fp32_mb:.1f} MB, {fp32_mb/quant_mb:.1f}x)")
print(f"  Inference: {infer_ms:.0f} ms/forecast")
print(f"  Power:     {avg_power:.1f} W")
print("Done.")
