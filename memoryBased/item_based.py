"""Item-based kNN recommender using adjusted-cosine similarity.

Implementation notes:
- Uses adjusted cosine (subtract user means) to compute item-item similarity.
- Predicts using weighted sum of user's ratings on neighbor items.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('item_based')
class ItemBasedRecommender:
    def __init__(self, k: int = 30):
        self.k = int(k)
        self.user_index: Dict[int, int] = {}
        self.index_user: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}
        self.R = None
        self.mask = None
        self.user_means = None
        self.item_means = None
        self.global_mean = 3.0
        self.sim = None  # item-item similarity matrix

    def fit(self, train_df: pd.DataFrame):
        users = sorted(train_df['user_id'].unique())
        items = sorted(train_df['item_id'].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.index_user = {i: u for u, i in self.user_index.items()}
        self.item_index = {it: j for j, it in enumerate(items)}
        self.index_item = {j: it for it, j in self.item_index.items()}

        n_users = len(users)
        n_items = len(items)
        R = np.zeros((n_users, n_items), dtype=float)
        mask = np.zeros((n_users, n_items), dtype=bool)

        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            it = int(row['item_id'])
            r = float(row['rating'])
            ui = self.user_index[u]
            ij = self.item_index[it]
            R[ui, ij] = r
            mask[ui, ij] = True

        self.R = R
        self.mask = mask

        user_sums = np.where(mask, R, 0.0).sum(axis=1)
        user_counts = mask.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.user_means = np.where(user_counts > 0, user_sums / user_counts, 0.0)

        item_sums = np.where(mask, R, 0.0).sum(axis=0)
        item_counts = mask.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.item_means = np.where(item_counts > 0, item_sums / item_counts, 0.0)

        all_ratings = R[mask]
        self.global_mean = float(np.mean(all_ratings)) if all_ratings.size > 0 else 3.0

        # adjusted ratings (subtract user means)
        adjusted = np.zeros_like(R)
        for i in range(n_users):
            adjusted[i, mask[i]] = R[i, mask[i]] - self.user_means[i]

        # compute item-item similarity (adjusted cosine)
        # items x items matrix = adjusted.T dot adjusted
        item_vecs = adjusted.T  # shape n_items x n_users
        norms = np.linalg.norm(item_vecs, axis=1)
        norms[norms == 0] = 1e-9
        sim = item_vecs.dot(item_vecs.T) / (norms[:, None] * norms[None, :])
        np.fill_diagonal(sim, -np.inf)
        self.sim = sim

    def predict(self, user: int, item: int) -> float:
        if self.R is None:
            return self.global_mean
        uidx = self.user_index.get(user, None)
        iidx = self.item_index.get(item, None)
        if uidx is None and iidx is None:
            return self.global_mean
        if uidx is None:
            return float(self.item_means[iidx]) if iidx is not None else self.global_mean
        if iidx is None:
            return float(self.user_means[uidx])

        # items rated by user
        user_rated = np.where(self.mask[uidx])[0]
        if user_rated.size == 0:
            return float(self.user_means[uidx])

        sims = self.sim[iidx, user_rated]
        # pick top-k neighbors
        top_k_idx = np.argsort(sims)[-self.k:][::-1]
        neighbors = user_rated[top_k_idx]
        weights = sims[top_k_idx]

        pos_mask = weights > 0
        if not np.any(pos_mask):
            return float(self.user_means[uidx])

        weights = weights[pos_mask]
        neighbors = neighbors[pos_mask]

        numer = 0.0
        denom = 0.0
        for w, j in zip(weights, neighbors):
            # use adjusted rating r_uj - user_mean[u]
            numer += w * (self.R[uidx, j] - self.user_means[uidx])
            denom += abs(w)

        if denom == 0:
            return float(self.user_means[uidx])

        pred = self.user_means[uidx] + (numer / denom)
        return float(pred)

    def recommend(self, user: int, k: int = 10) -> List[int]:
        if self.R is None:
            return []
        uidx = self.user_index.get(user, None)
        if uidx is None:
            sorted_idx = np.argsort(self.item_means)[::-1]
            return [int(self.index_item[i]) for i in sorted_idx[:k]]

        user_rated = self.mask[uidx]
        candidates = np.where(~user_rated)[0]
        scores = []
        for iidx in candidates:
            p = self.predict(int(user), int(self.index_item[iidx]))
            scores.append((iidx, p))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [int(self.index_item[i]) for i, _ in scores[:k]]
        return top
