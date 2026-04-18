# Architecture Overview

This project follows a package-first layout:

- `src/lightgcn_project/models`: LightGCN and baseline models
- `src/lightgcn_project/data`: data loading and BPR sampling
- `src/lightgcn_project/evaluation`: ranking metrics
- `scripts`: executable experiment entrypoints
- `outputs/tables`: generated result tables
- `data/raw`: expected local raw dataset location

Environment variable support:

- `LIGHTGCN_DATA_PATH`: override dataset location for all scripts
