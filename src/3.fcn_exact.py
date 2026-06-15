import matplotlib
matplotlib.use("Agg")
import argparse, os
import numpy as np
import torch, zarr
from datetime import datetime, timedelta
from earth2studio.data import GFS
from earth2studio.models.px import FCN
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

VARIABLES   = ["z500", "z300", "z700", "z1000", "t850", "t2m", "tcwv"]
EVAL_DATES  = ["2022-01-01","2022-04-01","2022-07-01","2022-10-01"]
NSTEPS      = 20
CALIB_DATE  = "2022-01-01"
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

parser = argparse.ArgumentParser()
parser.add_argument("--config",  required=True, choices=list(CONFIGS))
parser.add_argument("--outdir",  required=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gfs     = GFS()
package = FCN.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

# detect variables
_io = ZarrBackend()
_m  = FCN.load_model(package).to(device).eval()
with torch.no_grad():
    _io = run.deterministic([datetime.fromisoformat(CALIB_DATE)], 1, _m, gfs, _io)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
MODEL_VARS = [v for v in VARIABLES if v in avail(_io) and v not in coord_keys]
w_lat      = np.cos(np.deg2rad(_io["lat"][:])); w_lat = (w_lat/w_lat.mean())[np.newaxis,:,np.newaxis]
lat_rmse   = lambda a, b: np.sqrt(np.mean(w_lat*(a-b)**2, axis=(-1,-2)))
del _m
print(f"Config: {args.config}  Variables: {MODEL_VARS}")

# load and quantize
model = FCN.load_model(package).to(device).eval()
ci    = datetime.fromisoformat(CALIB_DATE)
if CONFIGS[args.config] is not None:
    def fwd(inner):
        model.model = inner
        with torch.no_grad():
            run.deterministic([ci], CALIB_STEPS, model, gfs, ZarrBackend())
    print(f"Calibrating {args.config} …")
    model.model = mtq.quantize(model.model, CONFIGS[args.config], fwd)

fp32_model  = FCN.load_model(package).to(device).eval()
probe_model = FCN.load_model(package).to(device).eval()

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

re, rf, r32, leads = {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}, {v: [] for v in MODEL_VARS}, None

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

slug = f"fcn_{args.config}"
np.save(os.path.join(args.outdir, f"{slug}_rmse_vs_gfs.npy"),
        {v: np.stack(re[v])  for v in MODEL_VARS if re[v]},  allow_pickle=True)
np.save(os.path.join(args.outdir, f"{slug}_rmse_vs_fp32.npy"),
        {v: np.stack(rf[v])  for v in MODEL_VARS if rf[v]},  allow_pickle=True)
np.save(os.path.join(args.outdir, f"{slug}_fp32_vs_gfs.npy"),
        {v: np.stack(r32[v]) for v in MODEL_VARS if r32[v]}, allow_pickle=True)
np.save(os.path.join(args.outdir, f"{slug}_leads.npy"), leads)
print(f"Saved results for {args.config}")
