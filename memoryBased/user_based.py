"""User-based kNN memory recommender.

Simple implementation:
- Builds a user-item rating matrix from training data.
- Mean-centers user rating vectors and computes cosine similarity.
- Predicts via weighted average of neighbor deviations from their means.

"""
from typing import Dict, List
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('user_based')
class UserBasedRecommender:
    def __init__(self, k: int = 30):
        self.k = int(k)
        self.user_index: Dict[int, int] = {}
        self.index_user: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}
        self.ratings_matrix = None  # centered ratings (users x items)
        self.rated_mask = None      # boolean matrix where ratings exist
        self.user_means = None
        self.item_means = None
        self.global_mean = 3.0

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

        # compute means
        user_sums = np.where(mask, R, 0.0).sum(axis=1)
        user_counts = mask.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            user_means = np.where(user_counts > 0, user_sums / user_counts, 0.0)

        item_sums = np.where(mask, R, 0.0).sum(axis=0)
        item_counts = mask.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            item_means = np.where(item_counts > 0, item_sums / item_counts, 0.0)

        all_ratings = R[mask]
        self.global_mean = float(np.mean(all_ratings)) if all_ratings.size > 0 else 3.0

        # center ratings by user means (only where mask True)
        centered = np.zeros_like(R)
        for i in range(n_users):
            if user_counts[i] > 0:
                centered[i, mask[i]] = R[i, mask[i]] - user_means[i]

        self.ratings_matrix = centered
        self.rated_mask = mask
        self.user_means = user_means
        self.item_means = item_means

        # Precompute user vector norms for cosine similarity
        norms = np.linalg.norm(self.ratings_matrix, axis=1)
        # avoid zeros
        norms[norms == 0] = 1e-9
        self._norms = norms

    def _user_similarity(self, user_idx: int) -> np.ndarray:
        # cosine similarity between user_idx and all users
        v = self.ratings_matrix[user_idx]
        sims = self.ratings_matrix.dot(v) / (self._norms * np.linalg.norm(v) if np.linalg.norm(v) != 0 else self._norms)
        # set self similarity to -inf so it won't be chosen among top-k
        sims[user_idx] = -np.inf
        return sims

    def predict(self, user: int, item: int) -> float:
        # fallbacks
        if self.ratings_matrix is None:
            return self.global_mean
        uidx = self.user_index.get(user, None)
        iidx = self.item_index.get(item, None)
        if uidx is None and iidx is None:
            return self.global_mean
        if uidx is None:
            # unknown user -> item mean or global
            return float(self.item_means[iidx]) if iidx is not None else self.global_mean
        if iidx is None:
            return float(self.user_means[uidx])

        # compute similarities
        sims = self._user_similarity(uidx)

        # consider only users who have rated the item
        voters = np.where(self.rated_mask[:, iidx])[0]
        if voters.size == 0:
            # no neighbor rated the item
            return float(self.user_means[uidx])

        # pick top-k neighbors among voters
        candidate_sims = sims[voters]
        top_k_idx = np.argsort(candidate_sims)[-self.k:][::-1]
        neighbors = voters[top_k_idx]
        weights = candidate_sims[top_k_idx]

        # only keep positive weights to avoid pulling away
        pos_mask = weights > 0
        if not np.any(pos_mask):
            return float(self.user_means[uidx])

        weights = weights[pos_mask]
        neighbors = neighbors[pos_mask]

        numer = 0.0
        denom = 0.0
        for w, n in zip(weights, neighbors):
            numer += w * self.ratings_matrix[n, iidx]
            denom += abs(w)

        if denom == 0:
            return float(self.user_means[uidx])

        pred = self.user_means[uidx] + (numer / denom)
        return float(pred)

    def recommend(self, user: int, k: int = 10) -> List[int]:
        if self.ratings_matrix is None:
            return []
        uidx = self.user_index.get(user, None)
        if uidx is None:
            # recommend global popular items (by item_means)
            sorted_idx = np.argsort(self.item_means)[::-1]
            return [int(self.index_item[i]) for i in sorted_idx[:k]]

        # candidate items are those user hasn't rated
        user_rated = self.rated_mask[uidx]
        candidates = np.where(~user_rated)[0]
        scores = []
        for iidx in candidates:
            p = self.predict(int(user), int(self.index_item[iidx]))
            scores.append((iidx, p))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [int(self.index_item[i]) for i, _ in scores[:k]]
        return top
