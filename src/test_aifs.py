# these should all hit the archive
AIFS_DATES = [
    "2026-04-01T00:00:00",
    "2026-05-01T00:00:00", 
    "2026-06-10T00:00:00",
    "2026-06-11T00:00:00",  # confirmed working before
]

# test each one
from earth2studio.data import IFS
from earth2studio.models.px import AIFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run

model = AIFS.load_model(AIFS.load_default_package())
data  = IFS()

for d in AIFS_DATES:
    try:
        io = ZarrBackend(f"outputs/aifs_test_{d[:10]}.zarr")
        run([d], 1, model, data, io)
        print(f"{d} ✓")
    except Exception as e:
        print(f"{d} ✗ {str(e)[:60]}")
