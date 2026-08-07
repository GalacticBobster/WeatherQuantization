"""
Snapshot maps at T+60h for t2m (°C) and tcwv (kg/m²) across quantization configs.
Adds "Truth" panel showing ERA5 (for DLWP) or GFS analysis (for FCN) at the same 
valid time as the forecast final step.

Usage: python map_t2m_v2.py {dlwp|fcn}
Produces:
  <MODEL>_t2m_maps.png
  <MODEL>_tcwv_maps.png
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
print(f"\n=== Truth ({truth_name} analysis at {valid_time}) ===")
truth = {}
try:
    # use a probe model for a 0-step forecast which just fetches truth
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
PANEL_ORDER = ["Truth", "FP32", "W8A32", "INT8_SMOOTHQUANT", "INT4_AWQ",
               "W8A8", "W4A32", "W2A32"]

def plot_variable(var_name, units, converter, cmap, vmin=None, vmax=None):
    all_panels = {}
    if var_name in truth:
        all_panels[f"{truth_name} truth"] = truth[var_name]
    for cfg in PANEL_ORDER[1:]:
        if cfg in fields and var_name in fields[cfg]:
            all_panels[cfg] = fields[cfg][var_name]
    
    if not all_panels:
        print(f"No {var_name} data"); return

    n     = len(all_panels)
    nrows = min(4, n)
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 4*nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])

    # global range for consistent color scale
    stacked = np.stack([converter(v) for v in all_panels.values()])
    v_min   = vmin if vmin is not None else float(np.nanpercentile(stacked, 2))
    v_max   = vmax if vmax is not None else float(np.nanpercentile(stacked, 98))
    norm    = mcolors.Normalize(vmin=v_min, vmax=v_max)

    for ax, (label, field) in zip(axes, all_panels.items()):
        data = converter(field)
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                            shading="auto", rasterized=True)
        # highlight truth panel
        title_kw = {"fontsize": 14, "fontweight": "bold"}
        if "truth" in label.lower():
            title_kw["color"] = "darkred"
            for spine in ax.spines.values():
                spine.set_edgecolor("darkred"); spine.set_linewidth(2)
        ax.set_title(label, **title_kw)
        ax.set_xlabel("Lon", fontsize=16); ax.set_ylabel("Lat", fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
    for ax in axes[n:]: ax.set_visible(False)

    cbar = fig.colorbar(
      im,
      ax=axes[:n].tolist(),
      location="right",
      fraction=0.025,   # thinner colorbar (default ~0.046)
      pad=0.02          # spacing between plots and colorbar
      )  

   cbar.set_label(f"{var_name} ({units})", fontsize=22, labelpad=12)
   cbar.ax.tick_params(labelsize=18)

    fig.suptitle(f"{MODEL_NAME.upper()}: {var_name} at {NSTEPS*6}h, "
                 f" Initial Condition: {INIT_DATE}",
                 fontweight="bold", fontsize=15)

    out = os.path.join(OUTDIR, f"{MODEL_NAME}_{var_name}_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

plot_variable("t2m",  "°C",     lambda f: f - 273.15, cmap="RdBu_r", vmin=-40, vmax=40)
plot_variable("tcwv", "kg/m²",  lambda f: f,          cmap="viridis", vmin=0,  vmax=60)

print("\nDone.")
