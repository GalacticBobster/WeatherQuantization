"""
Measure DLWP and FCN FLOPs from a real FP32 forward pass.
Uses fvcore.FlopCountAnalysis with a real input captured via forward hook.

Runs on GPU node, ~5 min for both models.
Output: prints values + writes flops_measured.txt for reference.

Usage: python measure_flops.py
"""
import os, sys
from datetime import datetime
import torch
from earth2studio.io import ZarrBackend
import earth2studio.run as run
from earth2studio.data import ARCO, GFS
from earth2studio.models.px.dlwp import DLWP
from earth2studio.models.px.fcn  import FCN
from fvcore.nn import FlopCountAnalysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

OUTDIR = "/glade/derecho/scratch/ananyo/WeatherQuantization/outputs_combined"
os.makedirs(OUTDIR, exist_ok=True)
NSTEPS = 10   # T+60h autoregressive rollout

# ── helper: capture the real input via forward hook ───────────
def measure(model_cls, data_source, probe_date):
    model = model_cls.load_model(model_cls.load_default_package()).to(device).eval()

    sample_input = [None]
    def _capture(module, inp, out):
        if sample_input[0] is None:
            sample_input[0] = inp[0].detach().clone()
    hook = model.model.register_forward_hook(_capture)

    with torch.no_grad():
        run.deterministic(
            [datetime.fromisoformat(probe_date)], 1, model, data_source, ZarrBackend()
        )
    hook.remove()

    x = sample_input[0]
    print(f"  Captured input shape: {tuple(x.shape)}")

    macs           = FlopCountAnalysis(model.model, x).total()
    flops_step     = 2 * macs                        # 1 MAC = 2 FLOPs
    params         = sum(p.numel() for p in model.model.parameters())
    flops_forecast = flops_step * NSTEPS
    return params, flops_step, flops_forecast, tuple(x.shape)

# ── DLWP ─────────────────────────────────────────────────────
print("\n=== DLWP ===")
dlwp = measure(DLWP, ARCO(), "2020-01-01")
p, s, f, shape = dlwp
print(f"  Params:               {p/1e6:>8.2f} M")
print(f"  FLOPs / step:         {s/1e9:>8.2f} G")
print(f"  FLOPs / 60h forecast: {f/1e9:>8.2f} G")

# ── FCN ──────────────────────────────────────────────────────
print("\n=== FCN ===")
fcn = measure(FCN, GFS(), "2022-01-01")
p2, s2, f2, shape2 = fcn
print(f"  Params:               {p2/1e6:>8.2f} M")
print(f"  FLOPs / step:         {s2/1e9:>8.2f} G")
print(f"  FLOPs / 60h forecast: {f2/1e9:>8.2f} G")

# ── save to file for later reference ──────────────────────────
out = os.path.join(OUTDIR, "flops_measured.txt")
with open(out, "w") as fh:
    fh.write(f"[DLWP]\nparams_M={p/1e6:.4f}\nflops_step_G={s/1e9:.4f}\nflops_forecast_G={f/1e9:.4f}\ninput_shape={shape}\n\n")
    fh.write(f"[FCN]\nparams_M={p2/1e6:.4f}\nflops_step_G={s2/1e9:.4f}\nflops_forecast_G={f2/1e9:.4f}\ninput_shape={shape2}\n")

print(f"\nSaved: {out}")
print(f"\nFCN / DLWP FLOPs ratio: {s2/s:.0f}×")
