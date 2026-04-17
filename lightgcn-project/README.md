# LightGCN — Simplifying and Powering Graph Convolution Network for Recommendation

Implementation of the **LightGCN** paper (SIGIR 2020) for the Collaborative Filtering course project at IIITD.

**Paper:** [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)

## Overview
LightGCN strips away the heavy neural network operations (non-linear activations and feature transformations) of traditional Graph Convolutional Networks (like NGCF). It relies entirely on linear neighborhood aggregation on the user-item bipartite graph, proving that straightforward embedding propagation is highly effective for Collaborative Filtering.

## Setup Requirements
All data is treated as implicit feedback (binarized interactions). Evaluated on **Hit Rate (HR@K)** and **NDCG@K**.

## Project Team
- **P1:** Data & Eval Lead
- **P2:** SOTA Baselines Lead
- **P3:** Core LightGCN Arch
- **P4:** Training & Tuning

## Datasets
- MovieLens-1M
- Amazon-Book
