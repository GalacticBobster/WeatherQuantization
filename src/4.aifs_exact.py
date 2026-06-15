import matplotlib
matplotlib.use("Agg")
import argparse, os
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
from ecmwf.opendata import Client
from earth2studio.data import IFS
from earth2studio.models.px import AIFS
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

# AIFS minimum variables for the paper — t2m + geopotential levels
# will be filtered against what AIFS actually outputs
VARIABLES = ["z500", "z300", "z700", "z1000", "t2m", "t850"]

client = Client()

def get_latest_ifs_date():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1))
    return yesterday.strftime("%Y-%m-%dT00:00:00")


def make_cfg(wb=8, ab=None):
    return {"quant_cfg": [
        {"quantizer_name": "*",                 "enable": False},
        {"quantizer_name": "*weight_quantizer",  "cfg": {"num_bits": wb, "axis": 0}},
        *([] if ab is None else [
          {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": ab, "axis": None}}]),
        # non-parametric
        {"parent_class": "LayerNorm",        "quantizer_name": "*", "enable": False},
        {"parent_class": "GELU",             "quantizer_name": "*", "enable": False},
        {"parent_class": "SumAggregation",   "quantizer_name": "*", "enable": False},
        {"parent_class": "ReluBounding",     "quantizer_name": "*", "enable": False},
        {"parent_class": "HardtanhBounding", "quantizer_name": "*", "enable": False},
        {"parent_class": "FractionBounding", "quantizer_name": "*", "enable": False},
        # pre/post processors — normalizer must stay FP32
        {"quantizer_name": "*pre_processors*",  "enable": False},
        {"quantizer_name": "*post_processors*", "enable": False},
        {"quantizer_name": "*normalizer*",      "enable": False},
        {"quantizer_name": "*trainable*",       "enable": False},
        # encoder/decoder boundary layers
        {"quantizer_name": "*encoder.emb*",          "enable": False},
        {"quantizer_name": "*decoder.emb*",          "enable": False},
        {"quantizer_name": "*node_data_extractor*",  "enable": False},
    ], "algorithm": "max"}

CONFIGS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "INT8":             mtq.INT8_DEFAULT_CFG,
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

parser = argparse.ArgumentParser()
parser.add_argument("--config",      default="W8A32", choices=list(CONFIGS))
parser.add_argument("--calib_steps", type=int, default=8)
parser.add_argument("--calib_date",  default=get_latest_ifs_date())
parser.add_argument("--eval_dates",  nargs="+", default=[get_latest_ifs_date()])
parser.add_argument("--nsteps",      type=int, default=10)
parser.add_argument("--outdir",      default="outputs/aifs_ptq")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ifs     = IFS()
package = AIFS.load_default_package()
model   = AIFS.load_model(package).to(device).eval()

# AIFS wraps model differently — quantize model.model
calib_init = datetime.fromisoformat(args.calib_date)
_io = ZarrBackend()
with torch.no_grad():
    _io = run.deterministic([calib_init], 1, model, ifs, _io)
avail_vars  = list(zarr.open(_io.store, mode="r").keys())
coord_keys  = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS  = [v for v in VARIABLES if v in avail_vars and v not in coord_keys]
print(f"AIFS variables available: {MODEL_VARS}")

if args.config != "FP32":
    def forward_loop(m):
        model.model = m
        with torch.no_grad():
            run.deterministic([calib_init], args.calib_steps, model, ifs, ZarrBackend())
    print(f"Calibrating {args.config} ({args.calib_steps} steps) …")
    model.model = mtq.quantize(model.model, CONFIGS[args.config], forward_loop)
    mtq.print_quant_summary(model.model)

w_lat    = np.cos(np.deg2rad(_io["lat"][:]))
w_lat    = (w_lat / w_lat.mean())[np.newaxis, :, np.newaxis]
lat_rmse = lambda a, b: np.sqrt(np.mean(w_lat * (a - b) ** 2, axis=(-1, -2)))

def run_forecast(m, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([date_str], args.nsteps, m, ifs, io)
    av    = list(zarr.open(io.store, mode="r").keys())
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in MODEL_VARS if v in av}, leads

def get_ifs_truth(date_str, probe):
    # IFS verification: step 0 at each valid time = analysis field
    init, out = datetime.fromisoformat(date_str), {v: [] for v in MODEL_VARS}
    for step in range(args.nsteps + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=step*6)], 0, probe, ifs, io)
        av = list(zarr.open(io.store, mode="r").keys())
        for v in MODEL_VARS:
            if v in av: out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in MODEL_VARS if out[v]}

fp32_model  = AIFS.load_model(package).to(device).eval()
probe_model = AIFS.load_model(package).to(device).eval()
rmse_ifs_q, rmse_ifs_fp32, rmse_fp32 = [{v: [] for v in MODEL_VARS} for _ in range(3)]

for d in args.eval_dates:
    print(f"Evaluating {d} …")
    truth         = get_ifs_truth(d, probe_model)
    q_fore, leads = run_forecast(model,      d)
    f_fore, _     = run_forecast(fp32_model, d)
    for v in MODEL_VARS:
        if v not in truth: continue
        rmse_ifs_q[v].append(lat_rmse(q_fore[v], truth[v]))
        rmse_ifs_fp32[v].append(lat_rmse(f_fore[v], truth[v]))
        rmse_fp32[v].append(lat_rmse(q_fore[v], f_fore[v]))

agg   = lambda lst: (np.mean(lst, 0), np.std(lst, 0))
ncols = min(3, len(MODEL_VARS))
nrows = int(np.ceil(len(MODEL_VARS) / ncols))
slug  = f"aifs_{args.config}_c{args.calib_steps}"

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
axes = np.array(axes).flatten()
for ax, v in zip(axes, MODEL_VARS):
    if not rmse_ifs_q[v]: ax.set_visible(False); continue
    for data, col, lbl in [(rmse_ifs_q[v],"firebrick",args.config),
                            (rmse_ifs_fp32[v],"steelblue","FP32")]:
        m, s = agg(data)
        ax.plot(leads, m, color=col, lw=2, label=lbl)
        ax.fill_between(leads, m-s, m+s, color=col, alpha=0.15)
    ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Lat-wtd RMSE vs IFS analysis"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for ax in axes[len(MODEL_VARS):]: ax.set_visible(False)
fig.suptitle(f"AIFS RMSE vs IFS analysis — {args.config} vs FP32 | calib={args.calib_steps}steps "
             f"| {len(args.eval_dates)} dates | mean ±1σ", fontweight="bold")
fig.savefig(os.path.join(args.outdir, f"{slug}_rmse_vs_ifs.png"), dpi=150)

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
axes2 = np.array(axes2).flatten()
for ax, v in zip(axes2, MODEL_VARS):
    if not rmse_fp32[v]: ax.set_visible(False); continue
    m, s = agg(rmse_fp32[v])
    ax.plot(leads, m, color="darkorange", lw=2)
    ax.fill_between(leads, m-s, m+s, color="darkorange", alpha=0.15)
    ax.set_title(v, fontweight="bold"); ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Lat-wtd RMSE vs FP32"); ax.grid(True, alpha=0.3)
for ax in axes2[len(MODEL_VARS):]: ax.set_visible(False)
fig2.suptitle(f"AIFS RMSE vs FP32 — {args.config} | calib={args.calib_steps}steps "
              f"| {len(args.eval_dates)} dates | mean ±1σ", fontweight="bold")
fig2.savefig(os.path.join(args.outdir, f"{slug}_rmse_vs_fp32.png"), dpi=150)
plt.close("all")
print("Done.")
