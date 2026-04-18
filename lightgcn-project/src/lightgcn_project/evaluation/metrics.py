import numpy as np


def hit_rate_at_k(ranked_items, test_items, k):
    """
    Hit Rate @ K for one user.

    Inputs:
    - ranked_items: item indices sorted by predicted score (descending)
    - test_items: set of relevant items for this user
    - k: evaluation cutoff

    Returns:
    - 1.0 if at least one relevant item appears in top-k
    - 0.0 otherwise
    """
    top_k = set(ranked_items[:k].tolist())
    return 1.0 if len(top_k & test_items) > 0 else 0.0


def ndcg_at_k(ranked_items, test_items, k):
    """
    NDCG @ K with binary relevance for one user.

    DCG rewards placing relevant items higher in ranking.
    IDCG is the best possible DCG for this user at cutoff k.
    Output is normalized to [0, 1].
    """
    top_k = ranked_items[:k]
    # Discounted cumulative gain based on ranked relevant hits.
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in test_items)
    # Ideal gain if all relevant items were ranked first.
    n_relevant = min(len(test_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0
