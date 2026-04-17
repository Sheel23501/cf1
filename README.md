# CF1 Recommender Systems Project

This repository contains a LightGCN-based collaborative filtering project with baseline comparisons and ablation experiments.

Main project directory: [lightgcn-project](lightgcn-project)

## What This Repo Includes

- LightGCN implementation for implicit-feedback recommendation
- Baselines: MostPop, ItemKNN, and BPR-MF
- Evaluation with HR@K and NDCG@K
- Experiment scripts for:
	- training LightGCN
	- running all baselines
	- running ablation studies

## Repository Layout

```
.
├── README.md
├── requirements.txt
└── lightgcn-project/
		├── README.md
		├── requirements.txt
		├── scripts/
		├── src/
		└── results/
```

## Quick Start

1. Move into the project directory:

```bash
cd lightgcn-project
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your dataset where scripts expect it (default):

```text
data/movielens-1m/ml-100k 4/u.data
```

If your data path differs, update the filepath in:
- [lightgcn-project/scripts/train_lightgcn.py](lightgcn-project/scripts/train_lightgcn.py)
- [lightgcn-project/scripts/run_all_baselines.py](lightgcn-project/scripts/run_all_baselines.py)
- [lightgcn-project/scripts/ablation_study.py](lightgcn-project/scripts/ablation_study.py)

## Run Experiments

Train only LightGCN:

```bash
python scripts/train_lightgcn.py
```

Run baseline comparison table:

```bash
python scripts/run_all_baselines.py
```

Run ablation study:

```bash
python scripts/ablation_study.py
```

## Outputs

Generated result tables are saved in:
- [lightgcn-project/results/tables/comparison_ml100k.csv](lightgcn-project/results/tables/comparison_ml100k.csv)
- [lightgcn-project/results/tables/ablation_ml100k.csv](lightgcn-project/results/tables/ablation_ml100k.csv)

## Notes

- Root [requirements.txt](requirements.txt) is separate from the LightGCN experiment dependencies.
- For project-specific setup, use [lightgcn-project/requirements.txt](lightgcn-project/requirements.txt).
