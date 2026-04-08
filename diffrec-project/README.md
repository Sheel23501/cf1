# 🔬 DiffRec — Diffusion Recommender Model

Implementation of the paper **"Diffusion Recommender Model"** (SIGIR 2023) for the Collaborative Filtering course at IIITD.

**Paper:** [arXiv:2304.04971](https://arxiv.org/abs/2304.04971)  
**Official Code:** [YiyanXu/DiffRec](https://github.com/YiyanXu/DiffRec)

---

## 📖 Overview

DiffRec applies **diffusion models** to collaborative filtering. Instead of using GANs or VAEs, it models user preference generation as a **denoising process**:

1. **Forward Process:** Gradually add noise to a user's interaction vector
2. **Reverse Process:** Train an MLP to denoise — recovering the user's true preferences
3. **Recommendation:** Rank items by the denoised prediction scores

Key insight: Unlike image diffusion, DiffRec **does not corrupt to pure noise** — it preserves user signal.

## 🏗️ Project Structure

```
diffrec-project/
├── data/
│   ├── amazon-book/           # Primary dataset
│   │   ├── raw/               # Original downloaded files
│   │   └── processed/         # Binarized, train/val/test splits
│   └── yelp/                  # Secondary dataset
│       ├── raw/
│       └── processed/
├── src/
│   ├── data_loader.py         # Load → Binarize → Split
│   ├── metrics.py             # HR@K, NDCG@K evaluation
│   ├── diffusion/
│   │   ├── noise_schedule.py  # Linear, Cosine β schedules
│   │   ├── forward.py         # q(xₜ | x₀) — add noise
│   │   ├── reverse.py         # p(x₀ | xₜ) — denoise
│   │   └── denoiser.py        # MLP denoiser network
│   ├── models/
│   │   ├── diffrec.py         # Full DiffRec model
│   │   └── l_diffrec.py       # Latent-space DiffRec
│   └── baselines/
│       ├── mostpop.py         # Most Popular baseline
│       ├── itemknn.py         # Item-based KNN
│       ├── bpr_mf.py          # BPR Matrix Factorization
│       ├── lightgcn.py        # LightGCN
│       └── multvae.py         # Mult-VAE
├── scripts/
│   ├── train_diffrec.py       # Train DiffRec / L-DiffRec
│   ├── train_baselines.py     # Train all baselines
│   ├── evaluate.py            # Run evaluation on test set
│   └── ablation.py            # Run ablation experiments
├── results/
│   ├── tables/                # Saved CSV result tables
│   └── figures/               # Plots for video demo
├── reference/                 # Official DiffRec code (read-only reference)
├── requirements.txt
└── README.md
```

## 📦 Datasets

| Dataset | Users | Items | Interactions | Source |
|---------|-------|-------|-------------|--------|
| Amazon-Book | 52,643 | 91,599 | 2,984,108 | [Amazon Reviews](https://jmcauley.ucsd.edu/data/amazon/) |
| Yelp | 31,668 | 38,048 | 1,561,406 | [Yelp Dataset](https://www.yelp.com/dataset) |

All ratings are **binarized** (interacted = 1, not interacted = 0).

## 📊 Metrics

- **HR@K** (Hit Rate at K): Did any test item appear in the top-K?
- **NDCG@K** (Normalized Discounted Cumulative Gain at K): Position-aware ranking quality

Evaluated at K = 5, 10, 20.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train DiffRec
python scripts/train_diffrec.py --dataset amazon-book --model diffrec

# Train baselines
python scripts/train_baselines.py --dataset amazon-book --model all

# Evaluate
python scripts/evaluate.py --dataset amazon-book --model diffrec

# Run ablations
python scripts/ablation.py --dataset amazon-book
```

## 👥 Team

| Member | Role |
|--------|------|
| P1 | Data & Evaluation Lead |
| P2 | Forward Diffusion Lead |
| P3 | Denoiser Model Lead |
| P4 | Baselines & Analysis Lead |

## 📚 References

- Wang et al., "Diffusion Recommender Model," SIGIR 2023. [arXiv](https://arxiv.org/abs/2304.04971)
- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020. [arXiv](https://arxiv.org/abs/2006.11239)
- He et al., "LightGCN: Simplifying and Powering GCN for Recommendation," SIGIR 2020. [arXiv](https://arxiv.org/abs/2002.02126)
- Liang et al., "Variational Autoencoders for Collaborative Filtering," WWW 2018. [arXiv](https://arxiv.org/abs/1802.05814)
