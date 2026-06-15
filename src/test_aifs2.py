import matplotlib
matplotlib.use("Agg")
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timezone, timedelta

# ── dynamic archive date ──────────────────────────────────────
today      = datetime.now(timezone.utc)
init_date  = (today - timedelta(days=40)).strftime("%Y-%m-%dT00:00:00")
print(f"Today:     {today.strftime('%Y-%m-%d')}")
print(f"Init date: {init_date}")

from earth2studio.models.px import AIFS2
from earth2studio.data import IFS
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import torch

os.makedirs("outputs", exist_ok=True)

# ── load model ────────────────────────────────────────────────
print("\nLoading AIFS2 …")
package = AIFS2.load_default_package()
model   = AIFS2.load_model(package)

print(f"Model type:       {type(model)}")
print(f"Inner model type: {type(getattr(model, 'model', None))}")

# ── architecture ──────────────────────────────────────────────
print("\n=== AIFS2 Architecture ===")
inner = getattr(model, "model", model)
for name, mod in inner.named_modules():
    if name:
        print(f"  {type(mod).__name__:35s}  {name}")

# ── parameter count ───────────────────────────────────────────
try:
    total = sum(p.numel() for p in inner.parameters())
    print(f"\nParameters: {total/1e6:.1f}M")
    print(f"FP32 size:  {total*4/1e6:.1f} MB")
except Exception as e:
    print(f"Could not count parameters: {e}")

# ── input/output coords ───────────────────────────────────────
try:
    in_coords  = model.input_coords()
    out_coords = model.output_coords(in_coords)
    print(f"\nInput variables:  {in_coords['variable'].tolist()[:10]} ...")
    print(f"Output variables: {out_coords['variable'].tolist()[:10]} ...")
    print(f"Lead times:       {out_coords['lead_time']}")
    print(f"Lat shape:        {in_coords['lat'].shape}")
    print(f"Lon shape:        {in_coords['lon'].shape}")
except Exception as e:
    print(f"Could not get coords: {e}")

# ── run forecast ──────────────────────────────────────────────
print(f"\nRunning AIFS2 forecast from {init_date} …")
data     = IFS()
zarr_out = "outputs/aifs2_forecast.zarr"
io       = ZarrBackend(zarr_out)

try:
    run.deterministic([init_date], 4, model, data, io)
    ds   = zarr.open(zarr_out, mode="r")
    keys = [k for k in ds.keys() if k not in {"lat","lon","lead_time","time","batch","ensemble"}]
    print(f"Output variables: {keys[:15]} ...")
    print(f"Forecast shape:   {ds[keys[0]].shape}")

    # ── quick t2m map ─────────────────────────────────────────
    if "t2m" in ds:
        lats  = ds["lat"][:]
        lons  = ds["lon"][:]
        field = ds["t2m"][0, 0, :, :]   # first lead time
        norm  = mcolors.Normalize(vmin=220, vmax=310)
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        im = ax.pcolormesh(lons, lats, field, cmap="RdBu_r", norm=norm, rasterized=True)
        fig.colorbar(im, ax=ax, label="2m Temperature (K)", shrink=0.8)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        ax.set_title(f"AIFS2 t2m — T+6h from {init_date}", fontweight="bold")
        fig.savefig("outputs/aifs2_t2m.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: outputs/aifs2_t2m.png")

except Exception as e:
    print(f"Forecast failed: {e}")
    print("Check IFS data availability for this date")
