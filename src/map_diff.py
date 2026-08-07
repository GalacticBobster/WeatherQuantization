"""
Snapshot maps at T+60h showing DIFFERENCES from truth for each PTQ config.
Two colorbars: one for absolute truth field, one for symmetric difference.

For each variable:
  - Top-left panel:  Truth (ERA5 or GFS analysis) — absolute colorbar
  - Other panels:    Config forecast MINUS truth — symmetric diff colorbar

Usage: python map_diff.py {dlwp|fcn}
Produces:
  <MODEL>_t2m_diff_maps.png
  <MODEL>_tcwv_diff_maps.png
"""
import matplotlib
matplotlib.use("Agg")
import argparse, os, sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch, zarr
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

MODEL_NAME = sys.argv[1].lower() if len(sys.argv) > 1 else "dlwp"

# ═══════════════════════════════════════════════════════════════
INIT_DATE   = "2020-01-01" if MODEL_NAME == "dlwp" else "2022-01-01"
NSTEPS      = 10
CALIB_STEPS = 8
OUTDIR      = f"/glade/derecho/scratch/ananyo/WeatherQuantization/outputs_combined/{MODEL_NAME}_maps"
# ═══════════════════════════════════════════════════════════════

os.makedirs(OUTDIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── model + data source ───────────────────────────────────────
if MODEL_NAME == "dlwp":
    from earth2studio.models.px.dlwp import DLWP as ModelCls
    from earth2studio.data import ARCO
    data_src   = ARCO()
    truth_name = "ERA5"
    def make_cfg(wb=8, ab=None):
        cfg = {"*": {"enable": False},
               "*weight_quantizer": {"num_bits": wb, "axis": 0}}
        if ab is not None:
            cfg["*input_quantizer"] = {"num_bits": ab, "axis": None}
        for lyr in ["equatorial_downsample.0.*","polar_downsample.0.*",
                    "equatorial_last.*","polar_last.*"]:
            cfg[lyr] = {"enable": False}
        return {"quant_cfg": cfg, "algorithm": "max"}
else:
    from earth2studio.models.px.fcn import FCN as ModelCls
    from earth2studio.data import GFS
    data_src   = GFS()
    truth_name = "GFS"
    def make_cfg(wb=8, ab=None):
        cfg = {"*": {"enable": False},
               "*weight_quantizer": {"num_bits": wb, "axis": 0},
               "*norm*": {"enable": False},
               "*pos_drop*": {"enable": False},
               "model.patch_embed.*": {"enable": False},
               "model.head.*": {"enable": False}}
        if ab is not None:
            cfg["*input_quantizer"] = {"num_bits": ab, "axis": None}
        return {"quant_cfg": cfg, "algorithm": "max"}

CONFIGS = {
    "FP32":             None,
    "W8A8":             make_cfg(8, 8),
    "W8A32":            make_cfg(8, None),
    "W4A32":            make_cfg(4, None),
    "W2A32":            make_cfg(2, None),
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

package = ModelCls.load_default_package()
avail   = lambda io: list(zarr.open(io.store, mode="r").keys())

# ── run each config, capture t2m + tcwv at T+60h ─────────────
fields   = {}
lats, lons = None, None

for cfg_name, cfg in CONFIGS.items():
    print(f"\n=== {cfg_name} ===")
    try:
        with torch.inference_mode(False):
            model = ModelCls.load_model(package).to(device).eval()
            for p in model.model.parameters(): p.data = p.data.clone()
            for b in model.model.buffers():    b.data = b.data.clone()
            if cfg is not None:
                def fwd(inner):
                    model.model = inner
                    with torch.inference_mode(False):
                        run.deterministic([datetime.fromisoformat(INIT_DATE)],
                                          CALIB_STEPS, model, data_src, ZarrBackend())
                print(f"  Calibrating …")
                model.model = mtq.quantize(model.model, cfg, fwd)

        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([datetime.fromisoformat(INIT_DATE)],
                                    NSTEPS, model, data_src, io)
        if lats is None:
            lats = io["lat"][:]; lons = io["lon"][:]
        keep = {}
        for v in ["t2m", "tcwv"]:
            if v in avail(io):
                keep[v] = io[v][0, -1]
        fields[cfg_name] = keep
        print(f"  Captured T+{NSTEPS*6}h: {list(keep.keys())}")
    except Exception as e:
        print(f"  Failed: {e}")

# ── fetch analysis truth at valid time (T+60h) ───────────────
valid_time = datetime.fromisoformat(INIT_DATE) + timedelta(hours=NSTEPS*6)
print(f"\n=== Truth ({truth_name} at {valid_time}) ===")
truth = {}
try:
    probe = ModelCls.load_model(package).to(device).eval()
    io_t  = ZarrBackend()
    with torch.no_grad():
        io_t = run.deterministic([valid_time], 0, probe, data_src, io_t)
    for v in ["t2m", "tcwv"]:
        if v in avail(io_t):
            truth[v] = io_t[v][0, 0]
    print(f"  Fetched: {list(truth.keys())}")
except Exception as e:
    print(f"  Truth fetch failed: {e}")

# ── plot ──────────────────────────────────────────────────────
PANEL_ORDER = ["FP32", "W8A32", "INT8_SMOOTHQUANT", "INT4_AWQ",
               "W8A8", "W4A32", "W2A32"]

def plot_variable(var_name, units, converter, cmap_truth, cmap_diff,
                   truth_range=None, diff_range=None):
    if var_name not in truth:
        print(f"No {var_name} truth"); return

    truth_field = converter(truth[var_name])
    diffs = {}
    for cfg in PANEL_ORDER:
        if cfg in fields and var_name in fields[cfg]:
            fc_field = converter(fields[cfg][var_name])
            diffs[cfg] = fc_field - truth_field

    if not diffs:
        print(f"No forecasts for {var_name}"); return

    n_diffs = len(diffs)
    n_total = 1 + n_diffs   # +1 for truth panel
    nrows   = 4
    ncols   = int(np.ceil(n_total / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 4*nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    # truth colorbar range
    if truth_range is not None:
        t_min, t_max = truth_range
    else:
        t_min = float(np.nanpercentile(truth_field, 2))
        t_max = float(np.nanpercentile(truth_field, 98))
    norm_truth = mcolors.Normalize(vmin=t_min, vmax=t_max)

    # diff colorbar range (symmetric around 0)
    all_diffs = np.stack(list(diffs.values()))
    if diff_range is not None:
        d_lim = diff_range
    else:
        d_lim = float(np.nanpercentile(np.abs(all_diffs), 98))
    norm_diff = mcolors.Normalize(vmin=-d_lim, vmax=d_lim)

    # panel 0: truth
    ax0 = axes[0]
    im_t = ax0.pcolormesh(lons, lats, truth_field, cmap=cmap_truth, norm=norm_truth,
                          shading="auto", rasterized=True)
    ax0.set_title(f"{truth_name} truth", fontweight="bold", fontsize=11, color="darkred")
    for spine in ax0.spines.values():
        spine.set_edgecolor("darkred"); spine.set_linewidth(2)
    ax0.set_xlabel("Lon", fontsize=9); ax0.set_ylabel("Lat", fontsize=9)

    # panels 1..N: config differences
    im_d = None
    for ax, (cfg, diff) in zip(axes[1:], diffs.items()):
        im_d = ax.pcolormesh(lons, lats, diff, cmap=cmap_diff, norm=norm_diff,
                              shading="auto", rasterized=True)
        ax.set_title(f"{cfg} − truth", fontweight="bold", fontsize=11)
        ax.set_xlabel("Lon", fontsize=9); ax.set_ylabel("Lat", fontsize=9)

    for ax in axes[n_total:]: ax.set_visible(False)

    # two colorbars: one for truth (leftmost panel), one for diffs (rest)
    cbar_t = fig.colorbar(im_t, ax=[ax0], shrink=0.9, aspect=20,
                           location="right", pad=0.02)
    cbar_t.set_label(f"{var_name} ({units})", fontsize=10)

    cbar_d = fig.colorbar(im_d, ax=axes[1:n_total].tolist(), shrink=0.9, aspect=25,
                           location="right", pad=0.02)
    cbar_d.set_label(f"Δ{var_name} ({units})", fontsize=10)

    fig.suptitle(f"{MODEL_NAME.upper()} — {var_name} at T+{NSTEPS*6}h  "
                 f"|  init {INIT_DATE}  |  valid {valid_time.date()}",
                 fontweight="bold", fontsize=13)

    out = os.path.join(OUTDIR, f"{MODEL_NAME}_{var_name}_diff_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# t2m: Kelvin → Celsius, diff in °C (equivalent to K)
plot_variable("t2m", "°C", lambda f: f - 273.15,
              cmap_truth="RdBu_r", cmap_diff="RdBu_r",
              truth_range=(-40, 40), diff_range=15)   # ±15°C for diff

# tcwv: kg/m², diff in kg/m²
plot_variable("tcwv", "kg/m²", lambda f: f,
              cmap_truth="viridis", cmap_diff="RdBu_r",
              truth_range=(0, 60), diff_range=25)     # ±25 kg/m² for diff

print("\nDone.")
