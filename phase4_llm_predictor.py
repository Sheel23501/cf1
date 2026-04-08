"""
Phase 4 — LLM-Enhanced Collaborative Filtering Recommender
===========================================================
Professor Requirements (from screenshot):
  1. Ratings binarized (liked = rating >= 4)
  2. Metrics: NDCG@K and Hit Rate@K
  3. Submit PDF with tables + YouTube demo link

What this file does:
  - Runs THREE models and compares them:
      Model 1: Popularity Baseline (non-personalized)
      Model 2: Item-based CF (Phase 3 best config)
      Model 3: LLM-Enhanced CF (your contribution)
  - Outputs two comparison tables: Hit Rate and NDCG
  - All results saved to results/phase4_results.txt
"""

from __future__ import annotations

import os
import re
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

# ── Groq LLM client ──────────────────────────────────────────────────────────
from groq import Groq

# SECURITY FIX: use environment variable instead of hardcoded key
# Set this before running: export GROQ_API_KEY="your_key_here"
# OR just paste your key in the fallback below for now

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── Your existing src imports ─────────────────────────────────────────────────
from src.data import MovieLens100KLoader, create_random_split
from src.cf import ItemBasedCF, EnhancedItemBasedCF


# =============================================================================
# SECTION 1 — EVALUATION METRICS (NDCG + Hit Rate)
# Professor explicitly requires these two metrics
# =============================================================================

def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Hit Rate@K = 1 if at least one relevant item appears in top-K, else 0.
    Averaged over all users to get the final Hit Rate.
    """
    top_k = set(recommended[:k])
    return 1.0 if len(top_k & relevant) > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K = measures ranking quality.
    A liked item at rank 1 scores higher than the same item at rank 10.
    Range: 0 (worst) to 1 (perfect).
    """
    top_k = recommended[:k]

    # DCG: give credit for each relevant item, discounted by rank
    dcg = sum(
        1.0 / np.log2(rank + 2)   # rank is 0-indexed, so +2
        for rank, item in enumerate(top_k)
        if item in relevant
    )

    # Ideal DCG: if all relevant items were at the very top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all_users(
    recommendations: dict,   # {user_id: [ranked list of item_ids]}
    test_liked: dict,         # {user_id: set of liked item_ids in test}
    k_values: list = [5, 10, 20]
) -> dict:
    """
    Computes Hit Rate@K and NDCG@K for all K values.
    Returns a dict of metric name -> score.
    """
    results = defaultdict(list)

    for user_id, ranked_items in recommendations.items():
        relevant = test_liked.get(user_id, set())
        if len(relevant) == 0:
            continue  # skip users with no liked items in test set

        for k in k_values:
            hr   = hit_rate_at_k(ranked_items, relevant, k)
            ndcg = ndcg_at_k(ranked_items, relevant, k)
            results[f"HitRate@{k}"].append(hr)
            results[f"NDCG@{k}"].append(ndcg)

    # Average across all users
    return {metric: np.mean(scores) for metric, scores in results.items()}


# =============================================================================
# SECTION 2 — DATA PREPARATION
# Binarize ratings as professor requires
# =============================================================================

def binarize_ratings(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """
    Convert 1-5 star ratings to binary liked/not-liked.
    liked = 1 if rating >= threshold (default: 4)
    liked = 0 if rating < threshold
    This is what the professor explicitly asked for.
    """
    df = df.copy()
    df['liked'] = (df['rating'] >= threshold).astype(int)
    return df


def build_test_liked(test_df: pd.DataFrame) -> dict:
    """
    Build a lookup dict: {user_id: set of item_ids the user liked in test}.
    Only includes items with liked=1.
    """
    liked_dict = defaultdict(set)
    liked_test = test_df[test_df['liked'] == 1]
    for _, row in liked_test.iterrows():
        liked_dict[int(row['user_id'])].add(int(row['item_id']))
    return dict(liked_dict)


# =============================================================================
# SECTION 3 — RECOMMENDATION GENERATION (Top-N lists)
# Each model produces a ranked list of item_ids per user
# =============================================================================

def get_candidate_items(user_id: int, train_df: pd.DataFrame,
                        all_items: list, n_candidates: int = 100) -> list:
    """
    Returns items NOT yet rated by this user in training.
    These are the candidates we will rank.
    """
    rated_by_user = set(
        train_df[train_df['user_id'] == user_id]['item_id'].tolist()
    )
    candidates = [item for item in all_items if item not in rated_by_user]
    return candidates


# ── Model 1: Popularity Baseline ─────────────────────────────────────────────

def popularity_recommendations(
    train_df: pd.DataFrame,
    test_users: list,
    all_items: list,
    k: int = 20
) -> dict:
    """
    Non-personalized baseline: recommend the globally most-liked items.
    Uses binarized 'liked' column to rank by popularity.
    """
    # Count how many users liked each item
    item_popularity = (
        train_df[train_df['liked'] == 1]
        .groupby('item_id')['liked']
        .sum()
        .sort_values(ascending=False)
    )
    popular_items = item_popularity.index.tolist()

    recommendations = {}
    for user_id in test_users:
        rated = set(train_df[train_df['user_id'] == user_id]['item_id'])
        # Recommend popular items the user hasn't seen
        recs = [item for item in popular_items if item not in rated][:k]
        recommendations[user_id] = recs

    return recommendations


# ── Model 2: Item-based CF ────────────────────────────────────────────────────

def cf_recommendations(
    model,
    train_df: pd.DataFrame,
    test_users: list,
    all_items: list,
    k: int = 20
) -> dict:
    """
    Generate top-K recommendations using a fitted CF model.
    For each user, score all unrated items and rank them.
    """
    recommendations = {}

    for user_id in test_users:
        candidates = get_candidate_items(user_id, train_df, all_items, n_candidates=200)

        # Build a mini dataframe of (user_id, item_id) pairs to predict
        candidate_df = pd.DataFrame({
            'user_id': [user_id] * len(candidates),
            'item_id': candidates
        })

        scores = model.predict(candidate_df)

        # Handle NaN scores — default to 0
        scores = np.nan_to_num(scores, nan=0.0)

        # Rank candidates by predicted score descending
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations[user_id] = [item for item, _ in ranked[:k]]

    return recommendations


# ── Model 3: LLM-Enhanced CF ──────────────────────────────────────────────────

def construct_user_profile(user_id: int, train_df: pd.DataFrame,
                           top_n: int = 5) -> str:
    """
    Build a text description of user taste from their training ratings.
    Now includes BOTH liked and disliked movies for richer LLM context.
    Fix for original bug: was only showing liked movies.
    """
    user_ratings = train_df[train_df['user_id'] == user_id].copy()

    # Top liked movies
    liked = user_ratings.nlargest(top_n, 'rating')
    liked_str = "\n".join(
        f"  - {row['title']} (rated {row['rating']}/5)"
        for _, row in liked.iterrows()
    )

    # Bottom disliked movies (show up to 3)
    disliked = user_ratings.nsmallest(3, 'rating')
    disliked_str = "\n".join(
        f"  - {row['title']} (rated {row['rating']}/5)"
        for _, row in disliked.iterrows()
        if row['rating'] <= 2
    )

    profile = f"MOVIES THIS USER LIKED:\n{liked_str}"
    if disliked_str:
        profile += f"\n\nMOVIES THIS USER DID NOT LIKE:\n{disliked_str}"

    return profile


def llm_score_candidates(
    user_id: int,
    candidates: list,
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    max_candidates: int = 20
) -> dict:
    """
    Ask the LLM to score a batch of candidate movies for a user.
    Returns: {item_id: score (0-1)}

    Key fixes applied vs original code:
    - Passes ACTUAL movie titles (not "Movie ID: 204")
    - Passes ACTUAL genres (not "Unknown")
    - Uses temperature=0.3 (not 0.0 which caused all-3.0 outputs)
    - Shows both liked AND disliked movies in user profile
    - Asks for binary liked/not-liked to match binarization
    """
    # Build title and genre lookup
    title_lookup = dict(zip(items_df['item_id'], items_df['title']))
    genre_lookup = dict(zip(items_df['item_id'],
                            items_df.get('genre', pd.Series(['Unknown'] * len(items_df)))))

    user_profile = construct_user_profile(user_id, train_df)

    # Limit candidates to avoid huge prompts
    candidates_to_score = candidates[:max_candidates]

    # Build movie list for the prompt
    movie_list = "\n".join(
        f"{i+1}. {title_lookup.get(item_id, f'Movie {item_id}')} "
        f"[{genre_lookup.get(item_id, 'Unknown')}]"
        for i, item_id in enumerate(candidates_to_score)
    )

    prompt = f"""You are a movie recommendation system.

USER PROFILE:
{user_profile}

TASK: Based on this user's taste, predict whether they would LIKE or DISLIKE each movie below.
A user LIKES a movie if they would rate it 4 or 5 out of 5.

MOVIES TO EVALUATE:
{movie_list}

OUTPUT FORMAT: Reply ONLY with a JSON object mapping each number to a score.
Score 1.0 = very likely to like, 0.5 = neutral, 0.0 = very likely to dislike.
Example: {{"1": 0.9, "2": 0.3, "3": 0.7}}

Be realistic — not all movies suit every user. Output ONLY the JSON."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie taste predictor. Output only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,   # FIX: was 0.0, causing all outputs to be identical
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        # Clean potential markdown code fences
        raw = re.sub(r"```json|```", "", raw).strip()

        scores_by_index = json.loads(raw)

        # Map back from index to item_id
        item_scores = {}
        for idx_str, score in scores_by_index.items():
            try:
                idx = int(idx_str) - 1  # convert back to 0-indexed
                if 0 <= idx < len(candidates_to_score):
                    item_id = candidates_to_score[idx]
                    item_scores[item_id] = float(score)
            except (ValueError, IndexError):
                continue

        return item_scores

    except Exception as e:
        print(f"  LLM scoring failed for user {user_id}: {e}")
        # Fallback: neutral score for all candidates
        return {item_id: 0.5 for item_id in candidates_to_score}


def llm_enhanced_recommendations(
    cf_model,
    train_df: pd.DataFrame,
    test_users: list,
    all_items: list,
    items_df: pd.DataFrame,
    k: int = 20,
    n_cf_candidates: int = 50,
    llm_rerank_top: int = 20,
    delay: float = 0.5
) -> dict:
    """
    Two-stage LLM-enhanced recommendation:
      Stage 1: CF model generates top-50 candidate items per user
      Stage 2: LLM re-ranks the top-20 candidates
      Final:   Return top-K from LLM-re-ranked list

    This is the research contribution — combining CF and LLM.
    """
    recommendations = {}
    total = len(test_users)

    print(f"  Generating LLM-enhanced recommendations for {total} users...")

    for i, user_id in enumerate(test_users):

        if i % 10 == 0:
            print(f"  Progress: {i}/{total} users...")

        # ── Stage 1: CF generates candidate pool ──
        candidates = get_candidate_items(user_id, train_df, all_items)

        candidate_df = pd.DataFrame({
            'user_id': [user_id] * len(candidates),
            'item_id': candidates
        })

        cf_scores = cf_model.predict(candidate_df)
        cf_scores = np.nan_to_num(cf_scores, nan=0.0)

        # Get top CF candidates to pass to LLM
        cf_ranked = sorted(
            zip(candidates, cf_scores),
            key=lambda x: x[1],
            reverse=True
        )
        top_cf_candidates = [item for item, _ in cf_ranked[:n_cf_candidates]]

        # ── Stage 2: LLM re-ranks the CF candidates ──
        llm_scores = llm_score_candidates(
            user_id=user_id,
            candidates=top_cf_candidates,
            train_df=train_df,
            items_df=items_df,
            max_candidates=llm_rerank_top
        )

        # Combine: CF score for items LLM didn't score, LLM score for others
        # Normalize CF scores to [0, 1] range
        max_cf = max(cf_scores) if max(cf_scores) > 0 else 1.0
        cf_score_map = {
            item: score / max_cf
            for item, score in zip(candidates, cf_scores)
        }

        # Final score = weighted combination
        # LLM score gets 60% weight, CF score gets 40%
        final_scores = {}
        for item in top_cf_candidates:
            llm_s = llm_scores.get(item, 0.5)
            cf_s  = cf_score_map.get(item, 0.0)
            final_scores[item] = 0.6 * llm_s + 0.4 * cf_s

        # Rank by final score
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations[user_id] = [item for item, _ in ranked[:k]]

        # Respect API rate limits
        time.sleep(delay)

    return recommendations


# =============================================================================
# SECTION 4 — MAIN EXPERIMENT RUNNER
# Runs all 3 models and prints comparison tables
# =============================================================================

def print_results_table(all_results: dict, k_values: list):
    """Print a clean comparison table of all models."""

    print("\n" + "=" * 65)
    print("RESULTS TABLE 1 — HIT RATE@K")
    print("=" * 65)
    header = f"{'Model':<30}" + "".join(f"  HR@{k:<6}" for k in k_values)
    print(header)
    print("-" * 65)
    for model_name, metrics in all_results.items():
        row = f"{model_name:<30}"
        for k in k_values:
            val = metrics.get(f"HitRate@{k}", 0.0)
            row += f"  {val:.4f}  "
        print(row)

    print("\n" + "=" * 65)
    print("RESULTS TABLE 2 — NDCG@K")
    print("=" * 65)
    header = f"{'Model':<30}" + "".join(f"  NDCG@{k:<4}" for k in k_values)
    print(header)
    print("-" * 65)
    for model_name, metrics in all_results.items():
        row = f"{model_name:<30}"
        for k in k_values:
            val = metrics.get(f"NDCG@{k}", 0.0)
            row += f"  {val:.4f}  "
        print(row)
    print("=" * 65)


def main():
    os.makedirs("results", exist_ok=True)
    K_VALUES = [5, 10, 20]

    print("=" * 65)
    print("PHASE 4: LLM-Enhanced Recommender System")
    print("Metrics: Hit Rate@K and NDCG@K (as required by professor)")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/6] Loading MovieLens 100K data...")
    loader  = MovieLens100KLoader()
    ratings = loader.load_full_ratings()
    items   = loader.load_items()

    # Merge titles into ratings for LLM user profiles
    if 'title' not in ratings.columns:
        ratings = ratings.merge(
            items[['item_id', 'title']], on='item_id', how='left'
        )

    # ── Binarize ratings (professor requirement #1) ────────────────
    print("[2/6] Binarizing ratings (liked = rating >= 4)...")
    ratings = binarize_ratings(ratings, threshold=4)
    print(f"  Total liked interactions : {ratings['liked'].sum()}")
    print(f"  Total interactions       : {len(ratings)}")
    print(f"  Like rate                : {ratings['liked'].mean():.1%}")

    # ── Train/test split ───────────────────────────────────────────
    print("[3/6] Splitting data (80/20)...")
    train, test = create_random_split(ratings, test_size=0.2, random_state=42)
    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")

    # Build ground truth: what each user liked in the test set
    test_liked = build_test_liked(test)
    test_users = list(test_liked.keys())  # only users who liked something
    all_items  = ratings['item_id'].unique().tolist()

    print(f"  Evaluating on {len(test_users)} users with liked items in test")

    # For demo / fast run, limit to 100 users
    # REMOVE THIS LINE for full evaluation
    test_users = test_users[:100]
    print(f"  (Limited to {len(test_users)} users for speed — remove limit for full eval)")

    all_results = {}

    # ── Model 1: Popularity Baseline ──────────────────────────────
    print("\n[4/6] Running Model 1: Popularity Baseline...")
    pop_recs = popularity_recommendations(train, test_users, all_items, k=max(K_VALUES))
    all_results["1. Popularity Baseline"] = evaluate_all_users(
        pop_recs, test_liked, K_VALUES
    )
    print("  Done.")

    # ── Model 2: Item-based CF (best config from Phase 3) ─────────
    print("\n[5/6] Running Model 2: Item-based CF (Phase 3 best config)...")
    cf_model = EnhancedItemBasedCF(
        k=20,
        similarity="cosine",
        min_common=2,
        shrinkage=0.0,
    )
    cf_model.fit(train)
    cf_recs = cf_recommendations(cf_model, train, test_users, all_items, k=max(K_VALUES))
    all_results["2. Item-based CF"] = evaluate_all_users(
        cf_recs, test_liked, K_VALUES
    )
    print("  Done.")

    # ── Model 3: LLM-Enhanced CF ───────────────────────────────────
    print("\n[6/6] Running Model 3: LLM-Enhanced CF (your contribution)...")
    print("  (This calls the Groq API — may take a few minutes)")
    llm_recs = llm_enhanced_recommendations(
        cf_model=cf_model,
        train_df=train,
        test_users=test_users,
        all_items=all_items,
        items_df=items,
        k=max(K_VALUES),
        n_cf_candidates=50,
        llm_rerank_top=20,
        delay=0.3
    )
    all_results["3. LLM-Enhanced CF (Ours)"] = evaluate_all_users(
        llm_recs, test_liked, K_VALUES
    )
    print("  Done.")

    # ── Print and save results ─────────────────────────────────────
    print_results_table(all_results, K_VALUES)

    # Save to file for your PDF
    results_path = "results/phase4_results.txt"
    with open(results_path, "w") as f:
        f.write("PHASE 4 RESULTS\n")
        f.write("=" * 65 + "\n\n")
        f.write("Binarization threshold : rating >= 4\n")
        f.write(f"Test users evaluated   : {len(test_users)}\n")
        f.write(f"K values               : {K_VALUES}\n\n")

        f.write("HIT RATE TABLE\n")
        f.write("-" * 65 + "\n")
        for model_name, metrics in all_results.items():
            f.write(f"{model_name}\n")
            for k in K_VALUES:
                val = metrics.get(f"HitRate@{k}", 0.0)
                f.write(f"  HitRate@{k}: {val:.4f}\n")
            f.write("\n")

        f.write("NDCG TABLE\n")
        f.write("-" * 65 + "\n")
        for model_name, metrics in all_results.items():
            f.write(f"{model_name}\n")
            for k in K_VALUES:
                val = metrics.get(f"NDCG@{k}", 0.0)
                f.write(f"  NDCG@{k}: {val:.4f}\n")
            f.write("\n")

    print(f"\n✅ Results saved to {results_path}")
    print("\nNext steps:")
    print("  1. Remove the test_users[:100] limit for full evaluation")
    print("  2. Copy results tables into your PDF")
    print("  3. Record YouTube demo showing this script running")
    print("  4. Submit PDF with tables + YouTube link")


if __name__ == "__main__":
    main()