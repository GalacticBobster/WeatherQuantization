"""
DLWP PTQ benchmark: skill (RMSE + ACC) and hardware metrics across configs.
Uses ERA5 (ARCO) for both initial conditions and truth verification.

Power is sampled continuously by a background NVML thread that runs only for the
duration of each inference, so the reported figure is a load average rather than
an instantaneous draw. Energy per forecast is integrated from the same samples.
"""
import matplotlib
matplotlib.use("Agg")
import argparse, os, csv, time, platform, threading
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
from earth2studio.data import ARCO, WB2Climatology
from earth2studio.models.px.dlwp import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

# ═══════════════════════════════════════════════════════════════
VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]
EVAL_DATES  = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]
NSTEPS      = 10
CALIB_STEPS = 8
MODEL_NAME  = "DLWP"
POWER_HZ    = 20.0      # NVML sampling rate during inference
IDLE_SECS   = 3.0       # idle baseline sampled before the timing loop
# ═══════════════════════════════════════════════════════════════

def make_cfg(wb=8, ab=None):
    quant_cfg = {
        "*":                {"enable": False},
        "*weight_quantizer": {"num_bits": wb, "axis": 0},
    }
    if ab is not None:
        quant_cfg["*input_quantizer"] = {"num_bits": ab, "axis": None}
    for lyr in ["equatorial_downsample.0.*", "polar_downsample.0.*",
                "equatorial_last.*", "polar_last.*"]:
        quant_cfg[lyr] = {"enable": False}
    return {"quant_cfg": quant_cfg, "algorithm": "max"}

CONFIGS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}
BITS = {"FP32":32,"W8A8":8,"W8A32":8,"W4A32":4,"W2A32":2,
        "INT8_SMOOTHQUANT":8,"INT4_AWQ":4}

# ── args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(CONFIGS))
parser.add_argument("--outdir", required=True)
parser.add_argument("--nruns", type=int, default=5)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

slug    = f"dlwp_{args.config}"
device  = torch.device("cuda")
era5    = ARCO()
clim    = WB2Climatology()
package = DLWP.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

# ── NVML power sampling ───────────────────────────────────────
try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False
    print("WARNING: pynvml not available — power metrics will be NaN")


def _nvml_handle():
    """Resolve the NVML handle for the GPU torch is actually using.

    Index matching is unreliable under CUDA_VISIBLE_DEVICES (Derecho nodes carry
    4 GPUs), so match on UUID where torch exposes it and fall back to index 0.
    """
    pynvml.nvmlInit()
    try:
        want = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        want = str(want).replace("GPU-", "").lower()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            if want and want in uuid.replace("GPU-", "").lower():
                return h
    except Exception:
        pass
    return pynvml.nvmlDeviceGetHandleByIndex(0)


class PowerMonitor:
    """Samples GPU power on a background thread for the lifetime of a `with` block.

    Collects (timestamp, watts) pairs at POWER_HZ. Energy is trapezoidally
    integrated over the samples, so it covers exactly the wrapped region.
    """

    def __init__(self, handle, hz=POWER_HZ):
        self.handle, self.interval = handle, 1.0 / hz
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _loop(self):
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((time.perf_counter(), w))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.handle is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        return False

    @property
    def watts(self):
        return np.array([w for _, w in self.samples], dtype=float)

    def energy_j(self):
        """Trapezoidal integral of power over the sampling window."""
        if len(self.samples) < 2:
            return float("nan")
        t = np.array([s for s, _ in self.samples])
        return float(np.trapz(self.watts, t))


def sample_idle(handle, secs=IDLE_SECS):
    """Baseline draw with no work queued, for reporting inference power net of idle."""
    if handle is None:
        return float("nan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with PowerMonitor(handle) as pm:
        time.sleep(secs)
    w = pm.watts
    return float(np.mean(w)) if w.size else float("nan")


# ── helpers ───────────────────────────────────────────────────
def gpu_info():
    if not torch.cuda.is_available(): return {"gpu_name":"CPU","gpu_mem_total_mb":0}
    p = torch.cuda.get_device_properties(0)
    return {"gpu_name": p.name, "gpu_mem_total_mb": p.total_memory/1e6}

def model_size_mb(m, bits): return sum(p.numel() for p in m.parameters())*bits/8/1e6

def n_params_m(m): return sum(p.numel() for p in m.parameters())/1e6


def benchmark(m, handle, n_runs=5):
    """Time and power-profile a full forecast, n_runs times.

    Note: ARCO fetches the initial condition inside run.deterministic, so the
    'infer' phase still contains network I/O. Pre-caching ICs locally would make
    these numbers compute-only.
    """
    init = datetime.fromisoformat(EVAL_DATES[0])

    # warmup — one full forecast to prime CUDA context, caches, and remote data
    with torch.no_grad():
        run.deterministic([init], NSTEPS, m, era5, ZarrBackend())
    if torch.cuda.is_available(): torch.cuda.synchronize()

    times, means, maxes, energies, n_samp = [], [], [], [], 0
    for i in range(n_runs):
        io = ZarrBackend()
        if torch.cuda.is_available(): torch.cuda.synchronize()

        with PowerMonitor(handle) as pm:
            t0 = time.perf_counter()
            with torch.no_grad():
                run.deterministic([init], NSTEPS, m, era5, io)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()

        w = pm.watts
        times.append(1000 * (t1 - t0))
        means.append(float(np.mean(w)) if w.size else float("nan"))
        maxes.append(float(np.max(w)) if w.size else float("nan"))
        energies.append(pm.energy_j())
        n_samp += w.size
        print(f"  run {i+1}: {times[-1]:8.1f} ms | "
              f"mean {means[-1]:6.1f} W | peak {maxes[-1]:6.1f} W | "
              f"{energies[-1]:8.1f} J | {w.size} samples")

    return {
        "ms_mean":   float(np.mean(times)),
        "ms_std":    float(np.std(times)),
        "power_mean":float(np.nanmean(means)),
        "power_std": float(np.nanstd(means)),
        "power_max": float(np.nanmax(maxes)) if len(maxes) else float("nan"),
        "energy_j":  float(np.nanmean(energies)),
        "n_samples": int(n_samp),
    }

# ── probe ─────────────────────────────────────────────────────
_probe = DLWP.load_model(package).to(device).eval()
_io    = ZarrBackend()
with torch.no_grad():
    _io = run.deterministic([datetime.fromisoformat(EVAL_DATES[0])], 1, _probe, era5, _io)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS = [v for v in VARIABLES if v in avail(_io) and v not in coord_keys]

# Weights normalised to mean 1, matching the np.mean() reduction below
# (WeatherBench-2 convention). Do NOT switch to w/w.sum() without also
# replacing np.mean with a sum over the latitude axis.
w_lat = np.cos(np.deg2rad(_io["lat"][:]))
w_lat = (w_lat / w_lat.mean())[np.newaxis, :, np.newaxis]

print(f"Variables: {MODEL_VARS}")
del _probe

def lat_rmse(a, b): return np.sqrt(np.mean(w_lat*(a-b)**2, axis=(-1,-2)))

def lat_acc(f, t, c):
    fa, ta = f - c, t - c
    num    = np.sum(fa*ta, axis=(-1,-2))
    denom  = np.sqrt(np.sum(fa**2, axis=(-1,-2)) *
                     np.sum(ta**2, axis=(-1,-2)))
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
    model = DLWP.load_model(package).to(device).eval()
    for p in model.model.parameters(): p.data = p.data.clone()
    for b in model.model.buffers():    b.data = b.data.clone()

    if CONFIGS[args.config] is not None:
        def fwd(inner):
            model.model = inner
            with torch.inference_mode(False):
                run.deterministic([calib_init], CALIB_STEPS, model, era5, ZarrBackend())
        print(f"Calibrating {args.config} …")
        model.model = mtq.quantize(model.model, CONFIGS[args.config], fwd)

# Load these FRESH from the package. Deep-copying `model` here would make the
# "FP32 baseline" a copy of the quantized model and drive RMSE-vs-FP32 to zero.
fp32_model  = DLWP.load_model(package).to(device).eval()
probe_model = DLWP.load_model(package).to(device).eval()

# ── hardware ──────────────────────────────────────────────────
hw       = gpu_info()
handle   = _nvml_handle() if (_NVML and torch.cuda.is_available()) else None
theo_mb  = model_size_mb(model.model, BITS[args.config])
fp32_mb  = model_size_mb(model.model, 32)

print("Sampling idle power baseline …")
idle_w = sample_idle(handle)
print(f"  idle: {idle_w:.1f} W")

print(f"Benchmarking inference ({args.nruns} runs, power @ {POWER_HZ:.0f} Hz) …")
bench = benchmark(model, handle, n_runs=args.nruns)

# ── hardware CSV early (survives any later crash) ─────────────
import modelopt
csv_path = os.path.join(args.outdir, "hardware_metrics.csv")
new      = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "model","config","gpu_name","gpu_mem_total_mb",
        "theoretical_size_mb","fp32_size_mb","compression_ratio","params_M",
        "inference_ms_per_forecast","inference_ms_std",
        "idle_power_w","mean_power_w","power_std_w","peak_power_w",
        "net_power_w","energy_j_per_forecast","n_power_samples","power_sample_hz",
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
        "params_M":                 f"{n_params_m(model.model):.2f}",
        "inference_ms_per_forecast":f"{bench['ms_mean']:.0f}",
        "inference_ms_std":         f"{bench['ms_std']:.0f}",
        "idle_power_w":             f"{idle_w:.1f}",
        "mean_power_w":             f"{bench['power_mean']:.1f}",
        "power_std_w":              f"{bench['power_std']:.1f}",
        "peak_power_w":             f"{bench['power_max']:.1f}",
        "net_power_w":              f"{bench['power_mean']-idle_w:.1f}",
        "energy_j_per_forecast":    f"{bench['energy_j']:.1f}",
        "n_power_samples":          bench["n_samples"],
        "power_sample_hz":          f"{POWER_HZ:.0f}",
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

# ── NPY save early (before plots) ─────────────────────────────
def safe(d): return {v: [a.tolist() for a in d[v]] for v in MODEL_VARS}
payload = {
    "vars":  MODEL_VARS,
    "leads": leads.tolist() if leads is not None else [],
    "ae": safe(ae), "af": safe(af),
    "re": safe(re), "rf": safe(rf), "r32": safe(r32),
    "weight_norm": "mean",   # so a reader knows which RMSE convention this is
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

try: plot_panel(re, r32, "Lat-wtd RMSE vs ERA5",  "rmse_vs_era5")
except Exception as e: print(f"RMSE plot failed: {e}")
try: plot_panel(rf, None, "Lat-wtd RMSE vs FP32", "rmse_vs_fp32")
except Exception as e: print(f"RMSE-FP32 plot failed: {e}")
try: plot_panel(ae, af,  "Lat-wtd ACC", "acc", hline=0.6, ylim=(-0.1, 1.05))
except Exception as e: print(f"ACC plot failed: {e}")

if _NVML and handle is not None:
    try: pynvml.nvmlShutdown()
    except Exception: pass

print(f"\nDone: {slug}")
print(f"  Size:   {theo_mb:.1f} MB ({fp32_mb/theo_mb:.1f}x)")
print(f"  Time:   {bench['ms_mean']:.0f} ± {bench['ms_std']:.0f} ms/forecast")
print(f"  Power:  {bench['power_mean']:.1f} ± {bench['power_std']:.1f} W "
      f"(idle {idle_w:.1f} W, peak {bench['power_max']:.1f} W)")
print(f"  Energy: {bench['energy_j']:.1f} J/forecast")
