# WeatherQuantization
A systematic study of numerical precision effects on AI weather model inference. We characterize forecast skill degradation across a complete precision ladder using Post Training Quantization (PTQ). We use NVIDIA Earth2Studio and ModelOptimizer to implement PTQ in two different AI emulators: 

- Deep Learning Weather Prediction (DLWP)
- FourCastNet (FCN)

## Directory
```text
WeatherQuantization/
├── src/
│   ├── dlwp.py
│   ├── fcn.py
│   ├── scripts/
│        ├── ...
│        └── ...
│
├── outputs/
│   ├── ACC results
│   └── RMSE results
│
├── .gitignore
└── README.md
```

The src directory contains the python codes to implement PTQ and run test for power consumption, inference time and memory footprint for each configuration. The subfolder scripts contains PBS scripts to run inferences on GPU. They have been written for NSF NCAR Casper HPC. outputs folder contains the root mean square error (RMSE) and anomaly correlation coefficient (ACC) for each PTQ experiment. We trace the evolution of forecast skill scores over a short-term forecast horizon.

## Citation 

## References
