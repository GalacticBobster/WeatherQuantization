import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch, zarr
from datetime import datetime
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

# ═══════════════════════════════════════════════════════════════
# ▶ SWAP MODEL HERE
# ═══════════════════════════════════════════════════════════════
from earth2studio.data import GFS
from earth2studio.models.px import FCN
MODEL_CLS  = FCN
DATA_SRC   = GFS()
MODEL_NAME = "FCN"
PLOT_DATE  = "2022-01-01"
CALIB_DATE = "2022-01-01"
# ═══════════════════════════════════════════════════════════════

LEAD_STEP   = 10
NSTEPS      = 10
CALIB_STEPS = 8
OUTDIR      = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs/maps"

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

EXPERIMENTS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

os.makedirs(OUTDIR, exist_ok=True)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
package = MODEL_CLS.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

def run_fcast(m):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(PLOT_DATE)], NSTEPS, m, DATA_SRC, io)
    return io

# get lats/lons from a probe run
_probe = MODEL_CLS.load_model(package).to(device).eval()
_io    = run_fcast(_probe)
lats, lons = _io["lat"][:], _io["lon"][:]
del _probe

# ── run all experiments ───────────────────────────────────────
fields = {}
ci     = datetime.fromisoformat(CALIB_DATE)

for exp, cfg in EXPERIMENTS.items():
    print(f"Running {exp} …")
    m = MODEL_CLS.load_model(package).to(device).eval()
    if cfg is not None:
        def fwd(inner, _m=m):
            _m.model = inner
            with torch.no_grad():
                run.deterministic([ci], CALIB_STEPS, _m, DATA_SRC, ZarrBackend())
        m.model = mtq.quantize(m.model, cfg, fwd)
    io = run_fcast(m)
    if "t2m" in avail(io):
        fields[exp] = io["t2m"][0, LEAD_STEP]
    del m

# ── plot ──────────────────────────────────────────────────────
panel_order = list(fields.keys())
ncols = 4
nrows = int(np.ceil(len(panel_order) / ncols))
norm  = mcolors.Normalize(vmin=220, vmax=310)

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows), constrained_layout=True)
axes = np.array(axes).flatten()

for ax, name in zip(axes, panel_order):
    d = fields[name]
    im = ax.pcolormesh(lons, lats, d, cmap="RdBu_r", norm=norm, rasterized=True)
    ax.set_title(name, fontweight="bold", fontsize=11)
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    if name != "FP32" and "FP32" in fields:
        rmse = float(np.sqrt(np.mean((d - fields["FP32"])**2)))
        ax.text(0.02, 0.04, f"RMSE={rmse:.2f}K", transform=ax.transAxes,
                fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=2))

for ax in axes[len(panel_order):]: ax.set_visible(False)
fig.colorbar(im, ax=axes[:len(panel_order)], orientation="horizontal",
             fraction=0.02, pad=0.02, label="2m Temperature (K)")
fig.suptitle(f"{MODEL_NAME} t2m — T+{LEAD_STEP*6}h from {PLOT_DATE} | all configs",
             fontweight="bold", fontsize=13)
out = os.path.join(OUTDIR, f"{MODEL_NAME.lower()}_map_t2m_{PLOT_DATE}_T{LEAD_STEP*6}h.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
