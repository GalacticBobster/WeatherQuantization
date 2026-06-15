import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import zarr, torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

MAP_VARS   = {"tcwv": ("Total Column Water Vapor", "BrBG",   [0,70]),
              "t2m":  ("2m Temperature",           "RdBu_r", [220,310])}
PLOT_DATE  = "2020-01-01"
LEAD_STEP  = 10          # T+60h
NSTEPS     = 10
CALIB_DATE = "2020-01-01"
CALIB_STEPS = 8
OUTDIR     = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs/combined"

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
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "W1A32":            make_cfg(1, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

os.makedirs(OUTDIR, exist_ok=True)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5    = ARCO()
package = DLWP.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

def run_fcast(model):
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([datetime.fromisoformat(PLOT_DATE)], NSTEPS, model, era5, io)
    lats = io["lat"][:]; lons = io["lon"][:]
    return {v: io[v][0, LEAD_STEP] for v in list(MAP_VARS) if v in avail(io)}, lats, lons

def get_era5_field(lats, lons, probe):
    vt = datetime.fromisoformat(PLOT_DATE) + timedelta(hours=LEAD_STEP*6)
    io = ZarrBackend()
    with torch.no_grad():
        io = run.deterministic([vt], 0, probe, era5, io)
    return {v: io[v][0, 0] for v in list(MAP_VARS) if v in avail(io)}

# ── Run all experiments ───────────────────────────────────────────────────────
probe   = DLWP.load_model(package).to(device).eval()
ci      = datetime.fromisoformat(CALIB_DATE)
forecasts = {}

for exp, cfg in EXPERIMENTS.items():
    print(f"Running {exp} …")
    m = DLWP.load_model(package).to(device).eval()
    if cfg is not None:
        def fwd(inner, _m=m):
            _m.model = inner
            with torch.no_grad():
                run.deterministic([ci], CALIB_STEPS, _m, era5, ZarrBackend())
        m.model = mtq.quantize(m.model, cfg, fwd)
    forecasts[exp], lats, lons = run_fcast(m)
    del m

era5_fields = get_era5_field(lats, lons, probe)

# ── Plot — one figure per variable ─────────────────────────────────────────
panel_order = ["ERA5", "FP32"] + [e for e in EXPERIMENTS if e not in ("FP32",)]
all_data    = {"ERA5": era5_fields, **{e: forecasts[e] for e in EXPERIMENTS}}

for var, (title, cmap, clim) in MAP_VARS.items():
    ncols = 4
    nrows = int(np.ceil(len(panel_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    for ax, name in zip(axes, panel_order):
        d = all_data[name].get(var)
        if d is None:
            ax.set_visible(False); continue
        im = ax.pcolormesh(lons, lats, d, cmap=cmap, norm=norm, rasterized=True)
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        if name not in ("ERA5",) and var in era5_fields:
            diff_rms = float(np.sqrt(np.mean((d - era5_fields[var])**2)))
            ax.text(0.02, 0.04, f"RMSE={diff_rms:.1f}", transform=ax.transAxes,
                    fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2))

    for ax in axes[len(panel_order):]: ax.set_visible(False)
    fig.colorbar(im, ax=axes[:len(panel_order)], orientation="horizontal",
                 fraction=0.02, pad=0.02, label=title)
    fig.suptitle(f"{title} — T+{LEAD_STEP*6}h valid from {PLOT_DATE} | all configs vs ERA5",
                 fontweight="bold", fontsize=13)
    out = os.path.join(OUTDIR, f"map_{var}_{PLOT_DATE}_T{LEAD_STEP*6}h.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

print("Done.")
