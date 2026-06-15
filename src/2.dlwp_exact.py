import argparse, os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import torch, zarr
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

VARIABLES = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]

def make_cfg(weight_bits=8, act_bits=None):
    return {
        "quant_cfg": [
            {"quantizer_name": "*",                "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": weight_bits, "axis": 0}},
            *([] if act_bits is None else [
              {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": act_bits, "axis": None}}]),
            {"quantizer_name": "equatorial_downsample.0.*", "enable": False},
            {"quantizer_name": "polar_downsample.0.*",      "enable": False},
            {"quantizer_name": "equatorial_last.*",         "enable": False},
            {"quantizer_name": "polar_last.*",              "enable": False},
        ],
        "algorithm": {"type": "percentile", "percentile": 99.9},
    }

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
parser.add_argument("--calib_date",  default="2020-01-01")
parser.add_argument("--eval_dates",  nargs="+",
                    default=["2020-01-01","2020-04-01","2020-07-01","2020-10-01"])
parser.add_argument("--nsteps",      type=int, default=10)
parser.add_argument("--outdir",      default="outputs/dlwp_ptq")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5    = ARCO()
package = DLWP.load_default_package()
model   = DLWP.load_model(package).to(device).eval()

calib_init = datetime.fromisoformat(args.calib_date)
if args.config != "FP32":
    def forward_loop(m):
        model.model = m
        with torch.no_grad():
            run.deterministic([calib_init], args.calib_steps, model, era5, ZarrBackend())
    print(f"Calibrating {args.config} ({args.calib_steps} steps) …")
    model.model = mtq.quantize(model.model, CONFIGS[args.config], forward_loop)
    mtq.print_quant_summary(model.model)

def get_lats():
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([calib_init], 1, model, era5, io)
    return io["lat"][:]

w_lat    = np.cos(np.deg2rad(get_lats()))
w_lat    = (w_lat / w_lat.mean())[np.newaxis, :, np.newaxis]
lat_rmse = lambda a, b: np.sqrt(np.mean(w_lat * (a - b) ** 2, axis=(-1, -2)))

def run_forecast(m, date_str):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(date_str)], args.nsteps, m, era5, io)
    avail = list(zarr.open(io.store, mode="r").keys())
    leads = io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
    return {v: io[v][0] for v in VARIABLES if v in avail}, leads

def get_era5_truth(date_str, probe):
    init, out = datetime.fromisoformat(date_str), {v: [] for v in VARIABLES}
    for step in range(args.nsteps + 1):
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init + timedelta(hours=step*6)], 0, probe, era5, io)
        avail = list(zarr.open(io.store, mode="r").keys())
        for v in VARIABLES:
            if v in avail: out[v].append(io[v][0, 0])
    return {v: np.stack(out[v], 0) for v in VARIABLES if out[v]}

fp32_model  = DLWP.load_model(package).to(device).eval()
probe_model = DLWP.load_model(package).to(device).eval()

rmse_era5_q, rmse_era5_fp32, rmse_fp32 = [{v: [] for v in VARIABLES} for _ in range(3)]

for d in args.eval_dates:
    print(f"Evaluating {d} …")
    truth         = get_era5_truth(d, probe_model)
    q_fore, leads = run_forecast(model,      d)
    f_fore, _     = run_forecast(fp32_model, d)
    for v in VARIABLES:
        if v not in truth: continue
        rmse_era5_q[v].append(lat_rmse(q_fore[v], truth[v]))
        rmse_era5_fp32[v].append(lat_rmse(f_fore[v], truth[v]))
        rmse_fp32[v].append(lat_rmse(q_fore[v], f_fore[v]))

agg = lambda lst: (np.mean(lst, 0), np.std(lst, 0))

ncols = min(4, len(VARIABLES))
nrows = int(np.ceil(len(VARIABLES) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
axes = np.array(axes).flatten()

for ax, v in zip(axes, VARIABLES):
    if not rmse_era5_q[v]: ax.set_visible(False); continue
    for data, col, lbl in [
        (rmse_era5_q[v],   "firebrick",  args.config),
        (rmse_era5_fp32[v],"steelblue",  "FP32"),
    ]:
        m, s = agg(data)
        ax.plot(leads, m, color=col, lw=2, label=lbl)
        ax.fill_between(leads, m-s, m+s, color=col, alpha=0.15)
    ax.set_title(v, fontweight="bold")
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Lat-wtd RMSE vs ERA5")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

for ax in axes[len(VARIABLES):]: ax.set_visible(False)
fig.suptitle(f"DLWP RMSE vs ERA5 — {args.config} vs FP32 | calib={args.calib_steps}steps "
             f"| {len(args.eval_dates)} dates | mean ±1σ", fontweight="bold")
slug = f"dlwp_{args.config}_c{args.calib_steps}"
fig.savefig(os.path.join(args.outdir, f"{slug}_rmse_vs_era5.png"), dpi=150)

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True)
axes2 = np.array(axes2).flatten()
for ax, v in zip(axes2, VARIABLES):
    if not rmse_fp32[v]: ax.set_visible(False); continue
    m, s = agg(rmse_fp32[v])
    ax.plot(leads, m, color="darkorange", lw=2)
    ax.fill_between(leads, m-s, m+s, color="darkorange", alpha=0.15)
    ax.set_title(v, fontweight="bold")
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Lat-wtd RMSE vs FP32")
    ax.grid(True, alpha=0.3)
for ax in axes2[len(VARIABLES):]: ax.set_visible(False)
fig2.suptitle(f"DLWP RMSE vs FP32 — {args.config} | calib={args.calib_steps}steps "
              f"| {len(args.eval_dates)} dates | mean ±1σ", fontweight="bold")
fig2.savefig(os.path.join(args.outdir, f"{slug}_rmse_vs_fp32.png"), dpi=150)
plt.close("all")
print("Done.")
