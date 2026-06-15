"""
dlwp_modelopt_quant.py  —  PTQ for Earth2Studio DLWP via NVIDIA ModelOpt
Usage: python dlwp_modelopt_quant.py --config INT8_SMOOTHQUANT
"""
import argparse, torch
from datetime import datetime
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import modelopt.torch.quantization as mtq

# ── Config map ────────────────────────────────────────────────────────────────
CONFIGS = {
    "FP8":              mtq.FP8_DEFAULT_CFG,
    "INT8":             mtq.INT8_DEFAULT_CFG,
    "INT8_SMOOTHQUANT": mtq.INT8_SMOOTHQUANT_CFG,
    "INT4_AWQ":         mtq.INT4_AWQ_CFG,
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="INT8_SMOOTHQUANT", choices=CONFIGS)
parser.add_argument("--calib_steps", type=int, default=4)   # × 6h steps
parser.add_argument("--calib_date",  default="2020-01-01")
args = parser.parse_args()

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5     = ARCO()
package  = DLWP.load_default_package()
model    = DLWP.load_model(package).to(device).eval()

# ── Calibration: run a short forecast and collect inner-model activations ─────
calib_init = datetime.fromisoformat(args.calib_date)

def forward_loop(m):
    """Feeds one short forecast through the inner torch.nn.Module for PTQ."""
    io = ZarrBackend()
    with torch.no_grad():
        # Temporarily swap model back so run.deterministic sees the real obj
        model.model = m
        run.deterministic([calib_init], args.calib_steps, model, era5, io)

print(f"Quantizing DLWP inner model with {args.config} …")
model.model = mtq.quantize(model.model, CONFIGS[args.config], forward_loop)
mtq.print_quant_summary(model.model)

# ── Inference test ────────────────────────────────────────────────────────────
io = ZarrBackend()
with torch.no_grad():
    io = run.deterministic([calib_init], 4, model, era5, io)

print("Quantized forecast complete. Lead times:", io["lead_time"][:])
