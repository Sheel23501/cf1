from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, k: int = 40) -> None:
        self.k = k
        self.global_mean: float | None = None
        self.user_means: pd.Series | None = None
        self.item_means: pd.Series | None = None
        self.user_item_matrix: pd.DataFrame | None = None
        self.item_similarity: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.global_mean = float(train_df["rating"].mean())

        self.user_item_matrix = train_df.pivot(
            index="user_id",
            columns="item_id",
            values="rating",
        )

        self.user_means = self.user_item_matrix.mean(axis=1)
        self.item_means = self.user_item_matrix.mean(axis=0)

        centered = self.user_item_matrix.subtract(self.item_means, axis=1)
        centered_filled = centered.fillna(0.0)

        similarity = cosine_similarity(centered_filled.T)
        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

    def _fallback_prediction(self, user_id: int, item_id: int) -> float:
        assert self.global_mean is not None
        assert self.user_means is not None
        assert self.item_means is not None

        user_mean = self.user_means.get(user_id, np.nan)
        item_mean = self.item_means.get(item_id, np.nan)

        if pd.notna(user_mean) and pd.notna(item_mean):
            pred = 0.5 * float(user_mean) + 0.5 * float(item_mean)
        elif pd.notna(user_mean):
            pred = float(user_mean)
        elif pd.notna(item_mean):
            pred = float(item_mean)
        else:
            pred = self.global_mean

        return float(np.clip(pred, 1.0, 5.0))

    def predict_one(self, user_id: int, item_id: int) -> float:
        assert self.user_item_matrix is not None
        assert self.item_similarity is not None
        assert self.item_means is not None

        if user_id not in self.user_item_matrix.index:
            return self._fallback_prediction(user_id, item_id)

        if item_id not in self.item_similarity.index:
            return self._fallback_prediction(user_id, item_id)

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        if user_ratings.empty:
            return self._fallback_prediction(user_id, item_id)

        rated_item_ids = user_ratings.index
        sims = self.item_similarity.loc[item_id, rated_item_ids]

        # Remove the target item itself and keep positively correlated neighbors.
        sims = sims[(sims.index != item_id) & (sims > 0)]
        if sims.empty:
            return self._fallback_prediction(user_id, item_id)

        top_neighbors = sims.sort_values(ascending=False).head(self.k)
        neighbor_ratings = user_ratings.loc[top_neighbors.index]
        neighbor_item_means = self.item_means.loc[top_neighbors.index]

        numerator = ((neighbor_ratings - neighbor_item_means) * top_neighbors).sum()
        denominator = np.abs(top_neighbors).sum()

        if denominator == 0:
            return self._fallback_prediction(user_id, item_id)

        target_item_mean = self.item_means.get(item_id, np.nan)
        if pd.isna(target_item_mean):
            return self._fallback_prediction(user_id, item_id)

        pred = float(target_item_mean + (numerator / denominator))
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = [
            self.predict_one(int(row.user_id), int(row.item_id))
            for row in test_df.itertuples(index=False)
        ]
        return np.array(preds, dtype=float)


class BiasBaseline:
    def __init__(self, reg: float = 10.0, n_iters: int = 15) -> None:
        self.reg = reg
        self.n_iters = n_iters
        self.mu: float = 0.0
        self.bu: pd.Series | None = None
        self.bi: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.mu = float(train_df["rating"].mean())

        users = sorted(train_df["user_id"].unique())
        items = sorted(train_df["item_id"].unique())
        self.bu = pd.Series(0.0, index=users)
        self.bi = pd.Series(0.0, index=items)

        user_groups = train_df.groupby("user_id")
        item_groups = train_df.groupby("item_id")

        for _ in range(self.n_iters):
            for user_id, group in user_groups:
                item_ids = group["item_id"].values
                ratings = group["rating"].values
                bi_vals = self.bi.loc[item_ids].values
                self.bu.loc[user_id] = float(
                    np.sum(ratings - self.mu - bi_vals) / (self.reg + len(group))
                )

            for item_id, group in item_groups:
                user_ids = group["user_id"].values
                ratings = group["rating"].values
                bu_vals = self.bu.loc[user_ids].values
                self.bi.loc[item_id] = float(
                    np.sum(ratings - self.mu - bu_vals) / (self.reg + len(group))
                )

    def predict(self, user_id: int, item_id: int) -> float:
        assert self.bu is not None
        assert self.bi is not None

        bu_val = float(self.bu.get(user_id, 0.0))
        bi_val = float(self.bi.get(item_id, 0.0))
        pred = self.mu + bu_val + bi_val
        return float(np.clip(pred, 1.0, 5.0))


class EnhancedItemBasedCF:
    def __init__(
        self,
        k: int = 80,
        similarity: str = "pearson",
        min_common: int = 3,
        shrinkage: float = 10.0,
    ) -> None:
        if similarity not in {"pearson", "cosine"}:
            raise ValueError("similarity must be 'pearson' or 'cosine'")

        self.k = k
        self.similarity = similarity
        self.min_common = min_common
        self.shrinkage = shrinkage

        self.user_item_matrix: pd.DataFrame | None = None
        self.item_similarity: pd.DataFrame | None = None
        self.bias_model = BiasBaseline(reg=10.0, n_iters=15)

    def _compute_common_counts(self, matrix: pd.DataFrame) -> pd.DataFrame:
        observed = matrix.notna().astype(int)
        counts = observed.T.dot(observed)
        return counts.astype(float)

    def _compute_item_similarity(self, matrix: pd.DataFrame) -> pd.DataFrame:
        common_counts = self._compute_common_counts(matrix)

        if self.similarity == "pearson":
            sim = matrix.corr(method="pearson", min_periods=self.min_common)
            sim = sim.fillna(0.0)
        else:
            centered = matrix.subtract(matrix.mean(axis=0), axis=1).fillna(0.0)
            sim_values = cosine_similarity(centered.T)
            sim = pd.DataFrame(sim_values, index=matrix.columns, columns=matrix.columns)
            sim = sim.where(common_counts >= self.min_common, 0.0)

        significance_weight = common_counts / (common_counts + self.shrinkage)
        sim = sim * significance_weight

        sim_values = sim.to_numpy(copy=True)
        np.fill_diagonal(sim_values, 0.0)
        sim = pd.DataFrame(sim_values, index=sim.index, columns=sim.columns)
        return sim

    def fit(self, train_df: pd.DataFrame) -> None:
        self.bias_model.fit(train_df)
        self.user_item_matrix = train_df.pivot(
            index="user_id",
            columns="item_id",
            values="rating",
        )
        self.item_similarity = self._compute_item_similarity(self.user_item_matrix)

    def predict_one(self, user_id: int, item_id: int) -> float:
        assert self.user_item_matrix is not None
        assert self.item_similarity is not None

        baseline_ui = self.bias_model.predict(user_id, item_id)

        if user_id not in self.user_item_matrix.index:
            return baseline_ui

        if item_id not in self.item_similarity.index:
            return baseline_ui

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        if user_ratings.empty:
            return baseline_ui

        sims = self.item_similarity.loc[item_id, user_ratings.index]
        sims = sims[np.isfinite(sims)]
        sims = sims[sims != 0.0]
        if sims.empty:
            return baseline_ui

        top_neighbors = sims.reindex(sims.abs().sort_values(ascending=False).index).head(self.k)
        neighbor_ratings = user_ratings.loc[top_neighbors.index]

        baseline_neighbors = np.array(
            [self.bias_model.predict(user_id, int(j)) for j in top_neighbors.index],
            dtype=float,
        )
        residuals = neighbor_ratings.values - baseline_neighbors
        finite_mask = np.isfinite(residuals) & np.isfinite(top_neighbors.values)
        if not np.any(finite_mask):
            return baseline_ui

        neighbor_weights = top_neighbors.values[finite_mask]
        residuals = residuals[finite_mask]

        numerator = float(np.dot(neighbor_weights, residuals))
        denominator = float(np.sum(np.abs(neighbor_weights)))

        if denominator == 0.0:
            return baseline_ui

        pred = baseline_ui + (numerator / denominator)
        if not np.isfinite(pred):
            return baseline_ui
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = [
            self.predict_one(int(row.user_id), int(row.item_id))
            for row in test_df.itertuples(index=False)
        ]
        return np.array(preds, dtype=float)
