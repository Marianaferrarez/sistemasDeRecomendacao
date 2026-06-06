"""Slope One memory-based recommender.

Implements the classic Slope One algorithm:
- Computes average deviations and counts for each item pair over training users.
- Predicts a user's rating for an item using the user's existing ratings and the pairwise deviations.

API: same minimal interface as other recommenders in `memoryBased/`.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('slope_one')
class SlopeOneRecommender:
    def __init__(self, shrink: float = 10.0, min_count: int = 1, rating_bounds=(1.0, 5.0)):
        # mapping ids to indices
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}
        # deviations matrix and counts
        self.deviations = None  # shape (n_items, n_items)
        self.counts = None
        self.item_means = None
        self.global_mean = 3.0
        # cache user->(item->rating)
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        # smoothing and thresholds
        self.shrink = float(shrink)
        self.min_count = int(min_count)
        self.rating_bounds = rating_bounds

    def fit(self, train_df: pd.DataFrame):
        items = sorted(train_df['item_id'].unique())
        self.item_index = {it: j for j, it in enumerate(items)}
        self.index_item = {j: it for it, j in self.item_index.items()}
        n_items = len(items)

        dev = np.zeros((n_items, n_items), dtype=float)
        cnt = np.zeros((n_items, n_items), dtype=int)

        self._user_ratings = {}
        user_groups = train_df.groupby('user_id')
        for user, user_df in user_groups:
            ratings = {int(r['item_id']): float(r['rating']) for _, r in user_df.iterrows()}
            self._user_ratings[int(user)] = ratings
            items_rated = list(ratings.keys())
            for i in range(len(items_rated)):
                it_i = items_rated[i]
                idx_i = self.item_index[it_i]
                r_i = ratings[it_i]
                for j in range(len(items_rated)):
                    if i == j:
                        continue
                    it_j = items_rated[j]
                    idx_j = self.item_index[it_j]
                    r_j = ratings[it_j]
                    dev[idx_i, idx_j] += (r_i - r_j)
                    cnt[idx_i, idx_j] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_dev = np.where(cnt > 0, dev / cnt, 0.0)

        self.deviations = avg_dev
        self.counts = cnt

        # item means and global mean
        item_sums = np.zeros(n_items, dtype=float)
        item_counts = np.zeros(n_items, dtype=int)
        for _, row in train_df.iterrows():
            it = int(row['item_id'])
            idx = self.item_index[it]
            item_sums[idx] += float(row['rating'])
            item_counts[idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.item_means = np.where(item_counts > 0, item_sums / item_counts, 0.0)

        all_ratings = train_df['rating'].astype(float).values
        self.global_mean = float(np.mean(all_ratings)) if all_ratings.size > 0 else 3.0

    def predict(self, user: int, item: int) -> float:
        # fallbacks
        if self.deviations is None:
            return self.global_mean

        iidx = self.item_index.get(item, None)
        if iidx is None:
            return self.global_mean

        user_ratings = self._user_ratings.get(int(user), None)
        if not user_ratings:
            return float(self.item_means[iidx]) if self.item_means is not None and self.item_means[iidx] > 0 else self.global_mean

        numer = 0.0
        denom = 0.0
        # for each item j the user rated, use deviation dev[i][j]
        for j_item, r_u_j in user_ratings.items():
            jidx = self.item_index.get(j_item, None)
            if jidx is None:
                continue
            c = int(self.counts[iidx, jidx])
            if c < self.min_count:
                continue
            # shrinkage weight: c / (c + shrink)
            weight = float(c) / (float(c) + self.shrink)
            numer += weight * (self.deviations[iidx, jidx] + r_u_j)
            denom += weight

        if denom == 0.0:
            fallback = float(self.item_means[iidx]) if self.item_means is not None and self.item_means[iidx] > 0 else self.global_mean
            return fallback

        pred = float(numer / denom)
        # clip to rating bounds
        low, high = self.rating_bounds
        if pred < low:
            pred = low
        elif pred > high:
            pred = high
        return pred

    def recommend(self, user: int, k: int = 10) -> List[int]:
        # if no model, nothing to recommend
        if self.deviations is None:
            return []

        user_ratings = self._user_ratings.get(int(user), {})

        # candidate items are those user hasn't rated
        all_idxs = set(range(len(self.index_item)))
        rated_idxs = set(self.item_index[it] for it in user_ratings.keys() if it in self.item_index)
        candidates = sorted(list(all_idxs - rated_idxs))

        scores = []
        for idx in candidates:
            it = self.index_item[idx]
            pred = self.predict(user, it)
            scores.append((it, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [int(it) for it, _ in scores[:k]]
        # if not enough candidates, fill with popular items
        if len(top) < k:
            popular = []
            if self.item_means is not None:
                popular = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1] if int(self.index_item[i]) not in user_ratings]
            for p in popular:
                if p not in top:
                    top.append(p)
                if len(top) >= k:
                    break
        return top[:k]
