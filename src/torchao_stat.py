from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import zarr
import os

# ═════════════════════════════════════════════════════════════════════════════
# ▶ ADD/REMOVE DATES HERE
# ═════════════════════════════════════════════════════════════════════════════

INIT_TIMES = [
    datetime(2020,  1,  1,  0),
    datetime(2020,  2,  1,  0),
    datetime(2020,  3,  1,  0),
    datetime(2020,  4,  1,  0),
    datetime(2020,  5,  1,  0),
    datetime(2020,  6,  1,  0),
    datetime(2020,  7,  1,  0),
    datetime(2020,  8,  1,  0),
    datetime(2020,  9,  1,  0),
    datetime(2020, 10,  1,  0),
    datetime(2020, 11,  1,  0),
    datetime(2020, 12,  1,  0),
]

# ▶ FORECAST LENGTH
NSTEPS = 10   # 6h steps → 60h total

# ▶ EXPERIMENTS — add/remove rows to change which precisions are tested
#   Each entry: (name, bits, quantization_fn applied to normalised activations)
#   All fns receive a FP32 tensor x with values in ~[-5, +6] (normalised space)

def _q(x, n_bits, sym=True):
    n   = 2 ** n_bits
    h   = n // 2 - 1
    if sym:
        sc = x.abs().max().clamp(min=1e-8) / h
        return torch.clamp(torch.round(x / sc), -h, h) * sc
    sc  = (x.max() - x.min()).clamp(min=1e-8) / (n - 1)
    xi  = torch.clamp(torch.round((x - x.min()) / sc), 0, n - 1)
    return xi * sc + x.min()

def _fp(x, eb, mb):
    native = {(4,3): getattr(torch,"float8_e4m3fn",None),
              (5,2): getattr(torch,"float8_e5m2",  None)}
    t = native.get((eb, mb))
    if t:
        try: return x.to(t).to(torch.float32)
        except Exception: pass
    zb   = 23 - mb
    if zb <= 0: return x
    mask = ~((1 << zb) - 1) & 0xFFFFFFFF
    out  = (x.float().view(torch.int32) & mask).view(torch.float32)
    if eb < 8:
        mv  = float(2 ** (2 ** (eb - 1)))
        out = torch.clamp(out, -mv, mv)
    return torch.where(torch.isnan(out)|torch.isinf(out),
                       torch.zeros_like(out), out)

EXPERIMENTS = [
    ("FP32",       32, lambda x: x),
    ("FP16",       16, lambda x: x.to(torch.float16).float()),
    ("BF16",       16, lambda x: x.to(torch.bfloat16).float()),
    ("FP8 E4M3",    8, lambda x: _fp(x, 4, 3)),
    ("INT8 asym",   8, lambda x: _q(x, 8, False)),
    ("INT8 sym",    8, lambda x: _q(x, 8, True)),
    ("FP4 E2M1",    4, lambda x: _fp(x, 2, 1)),
    ("INT4 asym",   4, lambda x: _q(x, 4, False)),
    ("INT4 sym",    4, lambda x: _q(x, 4, True)),
    ("FP2 E1M0",    2, lambda x: _fp(x, 1, 0)),
    ("INT2 sym",    2, lambda x: _q(x, 2, True)),
    ("INT1 sign",   1, lambda x: _q(x, 1, True)),
]

# ═════════════════════════════════════════════════════════════════════════════
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
print(f"Device : {device}")
print(f"Dates  : {len(INIT_TIMES)}")
print(f"Steps  : {NSTEPS} × 6h = {NSTEPS*6}h")
print(f"Expts  : {len(EXPERIMENTS)}")

# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ONCE
# ═════════════════════════════════════════════════════════════════════════════

package    = DLWP.load_default_package()
model_base = DLWP.load_model(package).to(device).eval()
era5       = ARCO()
orig_fwd   = model_base.model.forward

# Probe variables + coords from first date
_io_probe = ZarrBackend()
with torch.no_grad():
    _io_probe = run.deterministic([INIT_TIMES[0]], 1, model_base, era5, _io_probe)
coord_keys = {"lat","lon","lead_time","time","batch","ensemble"}
_z         = zarr.open(_io_probe.store, mode="r")
ALL_VARS   = [k for k in list(_z.keys()) if k not in coord_keys]
lats       = _io_probe["lat"][:]
lons       = _io_probe["lon"][:]
print(f"Variables: {ALL_VARS}")

w_lat = np.cos(np.deg2rad(lats))
w_lat = w_lat / w_lat.mean()
w_lat = w_lat[np.newaxis, :, np.newaxis]

def lat_rmse(a, b):
    return np.sqrt(np.mean(w_lat * (a - b)**2, axis=(-1,-2)))

def lat_acc(forecast, truth):
    clim   = truth.mean(axis=0, keepdims=True)
    fa, ta = forecast - clim, truth - clim
    num    = np.sum(w_lat * fa * ta,  axis=(-1,-2))
    denom  = np.sqrt(np.sum(w_lat*fa**2, axis=(-1,-2)) *
                     np.sum(w_lat*ta**2, axis=(-1,-2)))
    return np.where(denom > 0, num / denom, 0.0)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — iterate over dates then experiments
# ═════════════════════════════════════════════════════════════════════════════

# Accumulate per-date rows into a list → concatenate into DataFrame at end
all_rows = []

for date_idx, init_time in enumerate(INIT_TIMES):
    print(f"\n{'━'*70}")
    print(f"Date {date_idx+1}/{len(INIT_TIMES)}: {init_time}")
    print(f"{'━'*70}")

    # ── ERA5 verification at each lead time ──────────────────────────────────
    era5_truth = {var: [] for var in ALL_VARS}
    for step in range(NSTEPS + 1):
        vt       = init_time + timedelta(hours=step * 6)
        _io_v    = ZarrBackend()
        with torch.no_grad():
            _io_v = run.deterministic([vt], 0, model_base, era5, _io_v)
        for var in ALL_VARS:
            era5_truth[var].append(_io_v[var][0, 0])
    for var in ALL_VARS:
        era5_truth[var] = np.stack(era5_truth[var], axis=0)   # (nsteps+1, lat, lon)

    # ── Run each experiment ───────────────────────────────────────────────────
    exp_results = {}   # exp_name → {var: (nsteps+1, lat, lon)}

    for exp_name, bits, quant_fn in EXPERIMENTS:
        print(f"  {exp_name} ({bits}b)...", end=" ", flush=True)

        def make_patch(fn):
            def _p(*args, **kwargs):
                x   = args[0]
                x_q = fn(x)
                x_q = torch.where(torch.isnan(x_q)|torch.isinf(x_q),
                                   torch.zeros_like(x_q), x_q)
                out = orig_fwd(x_q, *args[1:], **kwargs)
                return out.float() if isinstance(out, torch.Tensor) else out
            return _p

        model_base.model.forward = make_patch(quant_fn)
        io = ZarrBackend()
        with torch.no_grad():
            io = run.deterministic([init_time], NSTEPS, model_base, era5, io)

        exp_results[exp_name] = {var: io[var][0] for var in ALL_VARS}
        print("done")

    model_base.model.forward = orig_fwd

    lead_hours = (io["lead_time"][:].astype("timedelta64[ns]")
                  .astype("timedelta64[h]").astype(int))

    # ── Compute skill scores ──────────────────────────────────────────────────
    fp32_ref = exp_results["FP32"]

    for exp_name, bits, _ in EXPERIMENTS:
        for var in ALL_VARS:
            fore  = exp_results[exp_name][var]    # (nsteps+1, lat, lon)
            truth = era5_truth[var]

            rmse_era5 = lat_rmse(fore, truth)     # (nsteps+1,)
            rmse_fp32 = lat_rmse(fore, fp32_ref[var])
            acc       = lat_acc(fore, truth)

            for i, h in enumerate(lead_hours):
                all_rows.append({
                    "init_time":    init_time.strftime("%Y-%m-%dT%H"),
                    "experiment":   exp_name,
                    "bits":         bits,
                    "variable":     var,
                    "lead_h":       int(h),
                    "rmse_vs_era5": float(rmse_era5[i]),
                    "rmse_vs_fp32": float(rmse_fp32[i]),
                    "acc_vs_era5":  float(acc[i]),
                })

# ═════════════════════════════════════════════════════════════════════════════
# BUILD DATAFRAME AND SAVE RAW CSV
# ═════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame(all_rows)
date_slug  = (f"{INIT_TIMES[0].strftime('%Y%m%d')}"
              f"_to_{INIT_TIMES[-1].strftime('%Y%m%d')}"
              f"_{len(INIT_TIMES)}dates")
raw_csv    = os.path.join(output_dir, f"skill_raw_{date_slug}.csv")
df.to_csv(raw_csv, index=False)
print(f"\nRaw CSV saved: {raw_csv}  ({len(df)} rows)")

# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATE STATISTICS ACROSS DATES
# ═════════════════════════════════════════════════════════════════════════════

agg = (df.groupby(["experiment","bits","variable","lead_h"])
         .agg(
             rmse_era5_mean  = ("rmse_vs_era5", "mean"),
             rmse_era5_std   = ("rmse_vs_era5", "std"),
             rmse_fp32_mean  = ("rmse_vs_fp32", "mean"),
             rmse_fp32_std   = ("rmse_vs_fp32", "std"),
             acc_mean        = ("acc_vs_era5",  "mean"),
             acc_std         = ("acc_vs_era5",  "std"),
             n_dates         = ("init_time",    "count"),
         )
         .reset_index()
)

agg_csv = os.path.join(output_dir, f"skill_aggregated_{date_slug}.csv")
agg.to_csv(agg_csv, index=False)
print(f"Aggregated CSV saved: {agg_csv}  ({len(agg)} rows)")

# ═════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE — T+60h, mean ± std across dates
# ═════════════════════════════════════════════════════════════════════════════

final_h  = NSTEPS * 6
bits_map = {e[0]: e[1] for e in EXPERIMENTS}

sub = agg[agg.lead_h == final_h].copy()

print(f"\n{'═'*110}")
print(f"RMSE vs ERA5 at T+{final_h}h — mean ± std across {len(INIT_TIMES)} dates")
print(f"{'Experiment':<16} {'Bits':>4}", end="")
for var in ALL_VARS:
    print(f"  {var:>14}", end="")
print()
print("─"*110)

for exp_name, bits, _ in EXPERIMENTS:
    row = sub[sub.experiment == exp_name]
    print(f"{exp_name:<16} {bits:>4}", end="")
    for var in ALL_VARS:
        r = row[row.variable == var]
        if len(r):
            m, s = float(r.rmse_era5_mean), float(r.rmse_era5_std)
            print(f"  {m:>6.1f}±{s:<5.1f}", end="")
        else:
            print(f"  {'N/A':>14}", end="")
    print()

print(f"\n{'═'*110}")
print(f"ACC vs ERA5 at T+{final_h}h — mean ± std across {len(INIT_TIMES)} dates")
print(f"{'Experiment':<16} {'Bits':>4}", end="")
for var in ALL_VARS:
    print(f"  {var:>14}", end="")
print()
print("─"*110)

for exp_name, bits, _ in EXPERIMENTS:
    row = sub[sub.experiment == exp_name]
    print(f"{exp_name:<16} {bits:>4}", end="")
    for var in ALL_VARS:
        r = row[row.variable == var]
        if len(r):
            m, s = float(r.acc_mean), float(r.acc_std)
            print(f"  {m:>6.3f}±{s:<5.3f}", end="")
        else:
            print(f"  {'N/A':>14}", end="")
    print()

print(f"\n{'═'*110}")
print(f"Quantization error (RMSE vs FP32) as % of FP32 forecast error — T+{final_h}h")
print(f"{'Experiment':<16} {'Bits':>4}", end="")
for var in ALL_VARS:
    print(f"  {var:>14}", end="")
print()
print("─"*110)

fp32_sub = sub[sub.experiment == "FP32"]
for exp_name, bits, _ in EXPERIMENTS:
    if exp_name == "FP32": continue
    row = sub[sub.experiment == exp_name]
    print(f"{exp_name:<16} {bits:>4}", end="")
    for var in ALL_VARS:
        r   = row[row.variable == var]
        fp  = fp32_sub[fp32_sub.variable == var]
        if len(r) and len(fp):
            pct_m = 100 * float(r.rmse_fp32_mean) / float(fp.rmse_era5_mean)
            pct_s = 100 * float(r.rmse_fp32_std)  / float(fp.rmse_era5_mean)
            print(f"  {pct_m:>6.1f}±{pct_s:<5.1f}%", end="")
        else:
            print(f"  {'N/A':>14}", end="")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════

colors_by_exp = {
    "FP32":      "black",
    "FP16":      "darkblue",
    "BF16":      "steelblue",
    "FP8 E4M3":  "cornflowerblue",
    "INT8 asym": "darkgreen",
    "INT8 sym":  "mediumseagreen",
    "FP4 E2M1":  "darkorange",
    "INT4 asym": "goldenrod",
    "INT4 sym":  "gold",
    "FP2 E1M0":  "firebrick",
    "INT2 sym":  "tomato",
    "INT1 sign": "lightcoral",
}
ls_map = {e[0]: ("-" if "FP" in e[0] or e[0]=="FP32" else "--")
          for e in EXPERIMENTS}

lead_vals = sorted(df.lead_h.unique())

# ── Figure 1: RMSE vs ERA5 per variable — mean with ±1σ shading ─────────────
ncols  = min(4, len(ALL_VARS))
nrows  = int(np.ceil(len(ALL_VARS) / ncols))
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                            constrained_layout=True)
axes1  = np.array(axes1).flatten()

for ax, var in zip(axes1, ALL_VARS):
    var_df = agg[agg.variable == var]
    for exp_name, bits, _ in EXPERIMENTS:
        ed  = var_df[var_df.experiment == exp_name].sort_values("lead_h")
        x   = ed.lead_h.values
        m   = ed.rmse_era5_mean.values
        s   = ed.rmse_era5_std.values
        col = colors_by_exp.get(exp_name, "gray")
        ls  = ls_map.get(exp_name, "-")
        ax.plot(x, m, color=col, ls=ls, lw=1.8,
                label=f"{exp_name}({bits}b)")
        ax.fill_between(x, m-s, m+s, color=col, alpha=0.10)
    ax.set_title(var, fontweight="bold")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE vs ERA5")
    ax.grid(True, alpha=0.3)

axes1[0].legend(fontsize=6, ncol=2)
for ax in axes1[len(ALL_VARS):]:
    ax.set_visible(False)

fig1.suptitle(
    f"RMSE vs ERA5 — mean ± 1σ across {len(INIT_TIMES)} dates\n"
    f"DLWP / ERA5  {INIT_TIMES[0].date()} to {INIT_TIMES[-1].date()}",
    fontsize=12, fontweight="bold"
)
p1 = os.path.join(output_dir, f"skill_rmse_era5_allvars_{date_slug}.png")
fig1.savefig(p1, dpi=150)
print(f"\nSaved: {p1}")

# ── Figure 2: ACC vs ERA5 per variable ───────────────────────────────────────
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                            constrained_layout=True)
axes2 = np.array(axes2).flatten()

for ax, var in zip(axes2, ALL_VARS):
    var_df = agg[agg.variable == var]
    for exp_name, bits, _ in EXPERIMENTS:
        ed  = var_df[var_df.experiment == exp_name].sort_values("lead_h")
        x   = ed.lead_h.values
        m   = ed.acc_mean.values
        s   = ed.acc_std.values
        col = colors_by_exp.get(exp_name, "gray")
        ls  = ls_map.get(exp_name, "-")
        ax.plot(x, m, color=col, ls=ls, lw=1.8,
                label=f"{exp_name}({bits}b)")
        ax.fill_between(x, m-s, m+s, color=col, alpha=0.10)
    ax.axhline(y=0.6, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_ylim([-0.1, 1.05])
    ax.set_title(var, fontweight="bold")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("ACC vs ERA5")
    ax.grid(True, alpha=0.3)

axes2[0].legend(fontsize=6, ncol=2)
for ax in axes2[len(ALL_VARS):]:
    ax.set_visible(False)

fig2.suptitle(
    f"ACC vs ERA5 — mean ± 1σ across {len(INIT_TIMES)} dates\n"
    f"DLWP / ERA5  {INIT_TIMES[0].date()} to {INIT_TIMES[-1].date()}",
    fontsize=12, fontweight="bold"
)
p2 = os.path.join(output_dir, f"skill_acc_era5_allvars_{date_slug}.png")
fig2.savefig(p2, dpi=150)
print(f"Saved: {p2}")

# ── Figure 3: Heatmap — mean RMSE vs ERA5 at T+60h ──────────────────────────
exp_names = [e[0] for e in EXPERIMENTS]
heatmap   = np.array([
    [float(sub[(sub.experiment==e)&(sub.variable==v)].rmse_era5_mean)
     if len(sub[(sub.experiment==e)&(sub.variable==v)]) else np.nan
     for v in ALL_VARS]
    for e in exp_names
])

fig3, ax3 = plt.subplots(figsize=(max(8, len(ALL_VARS)*1.4),
                                   max(5, len(exp_names)*0.7)),
                          constrained_layout=True)
valid = heatmap[~np.isnan(heatmap) & (heatmap > 0)]
im    = ax3.imshow(heatmap, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.LogNorm(vmin=valid.min(), vmax=valid.max()))
ax3.set_xticks(range(len(ALL_VARS)))
ax3.set_xticklabels(ALL_VARS, rotation=45, ha="right", fontsize=9)
ax3.set_yticks(range(len(exp_names)))
ax3.set_yticklabels([f"{e[0]} ({e[1]}b)" for e in EXPERIMENTS], fontsize=9)
ax3.set_title(
    f"Mean RMSE vs ERA5 at T+{final_h}h — {len(INIT_TIMES)} dates\n"
    f"Log scale — darker = worse",
    fontsize=11
)
fig3.colorbar(im, ax=ax3, label="Lat-wtd RMSE vs ERA5 (log)", shrink=0.8)
for i in range(len(exp_names)):
    for j in range(len(ALL_VARS)):
        val   = heatmap[i, j]
        if not np.isnan(val):
            col = "white" if val > np.percentile(valid, 70) else "black"
            ax3.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=6.5, color=col)
p3 = os.path.join(output_dir, f"skill_heatmap_era5_{date_slug}.png")
fig3.savefig(p3, dpi=150)
print(f"Saved: {p3}")

# ── Figure 4: Quant error % of FP32 forecast error — mean ± std ─────────────
key_vars = [v for v in ["z500","t850","t2m"] if v in ALL_VARS] or ALL_VARS[:3]

fig4, axes4 = plt.subplots(1, len(key_vars), figsize=(6*len(key_vars), 5),
                            constrained_layout=True)
if len(key_vars) == 1:
    axes4 = [axes4]

for ax, var in zip(axes4, key_vars):
    fp32_agg = agg[(agg.experiment=="FP32")&(agg.variable==var)
                   ].sort_values("lead_h")

    for exp_name, bits, _ in EXPERIMENTS:
        if exp_name == "FP32": continue
        ed      = agg[(agg.experiment==exp_name)&(agg.variable==var)
                      ].sort_values("lead_h")
        fp32_m  = fp32_agg.rmse_era5_mean.values
        x       = ed.lead_h.values
        pct_m   = 100 * ed.rmse_fp32_mean.values / np.where(fp32_m>0, fp32_m, np.nan)
        pct_s   = 100 * ed.rmse_fp32_std.values  / np.where(fp32_m>0, fp32_m, np.nan)
        col     = colors_by_exp.get(exp_name, "gray")
        ls      = ls_map.get(exp_name, "-")
        ax.plot(x, pct_m, color=col, ls=ls, lw=1.8,
                label=f"{exp_name}({bits}b)")
        ax.fill_between(x, pct_m-pct_s, pct_m+pct_s, color=col, alpha=0.10)

    for thresh, sty in [(1,"--"),(10,":"),(100,"-.")]:
        ax.axhline(y=thresh, color="gray", lw=0.8, ls=sty, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Quant error % of FP32 forecast error")
    ax.set_title(var, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)

fig4.suptitle(
    f"Quantization Error as % of FP32 Forecast Error — {len(INIT_TIMES)} dates\n"
    f"100% = quant error equals model's own forecast error vs ERA5",
    fontsize=11, fontweight="bold"
)
p4 = os.path.join(output_dir, f"skill_quant_pct_{date_slug}.png")
fig4.savefig(p4, dpi=150)
print(f"Saved: {p4}")

plt.show()
print(f"\nAll done.  {len(INIT_TIMES)} dates × {len(EXPERIMENTS)} experiments × "
      f"{len(ALL_VARS)} variables = {len(all_rows)} skill score rows.")
