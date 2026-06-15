import modelopt.torch.quantization as mtq

# Print what a default config actually looks like
import json
print(json.dumps(mtq.INT8_DEFAULT_CFG, indent=2, default=str))


import modelopt.torch.quantization.calib as calib
print(dir(calib))
