"""Factorization Machines (FM) recommender.

This FM implementation uses one-hot user and item features (concatenated) so it
reduces to a generalised matrix factorization but supports the FM formulation
and can be extended with more features later.

Training optimizes squared error with SGD and L2 regularization.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('fm')
class FMRecommender:
    def __init__(self, n_factors: int = 10, lr: float = 0.01, reg_w: float = 0.01, reg_v: float = 0.01, n_epochs: int = 10, verbose: bool = False, rating_bounds=(1.0,5.0)):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg_w = float(reg_w)
        self.reg_v = float(reg_v)
        self.n_epochs = int(n_epochs)
        self.verbose = bool(verbose)
        self.rating_bounds = rating_bounds

        self.user_index: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}

        self.n_features = 0
        self.w0 = 0.0
        self.w = None  # linear weights per feature
        self.V = None  # factor matrix (n_features x n_factors)

        # caches
        self._user_items: Dict[int, set] = {}
        self.item_means = None

    def _make_feat_indices(self, user: int, item: int) -> List[int]:
        # one-hot feature indices: user_idx, offset + item_idx
        uidx = self.user_index.get(user, None)
        iidx = self.item_index.get(item, None)
        if uidx is None or iidx is None:
            return []
        return [uidx, len(self.user_index) + iidx]

    def fit(self, train_df: pd.DataFrame):
        users = sorted(train_df['user_id'].unique())
        items = sorted(train_df['item_id'].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {it: j for j, it in enumerate(items)}
        self.index_item = {j: it for it, j in self.item_index.items()}

        n_users = len(self.user_index)
        n_items = len(self.item_index)
        self.n_features = n_users + n_items

        rng = np.random.RandomState(2026)
        self.w0 = 0.0
        self.w = np.zeros(self.n_features, dtype=float)
        self.V = 0.01 * rng.randn(self.n_features, self.n_factors)

        # build training triples and caches
        triples = []  # (user, item, rating, feat_indices)
        self._user_items = {}
        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            it = int(row['item_id'])
            r = float(row['rating'])
            if u not in self.user_index or it not in self.item_index:
                continue
            feats = self._make_feat_indices(u, it)
            triples.append((u, it, r, feats))
            self._user_items.setdefault(u, set()).add(it)

        # item means
        item_sums = np.zeros(n_items, dtype=float)
        item_counts = np.zeros(n_items, dtype=int)
        for _, row in train_df.iterrows():
            it = int(row['item_id'])
            if it in self.item_index:
                idx = self.item_index[it]
                item_sums[idx] += float(row['rating'])
                item_counts[idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.item_means = np.where(item_counts > 0, item_sums / item_counts, 0.0)

        # SGD
        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'FM epoch {epoch+1}/{self.n_epochs}')
            rng.shuffle(triples)
            for u, it, r, feats in triples:
                if not feats:
                    continue
                # compute linear term
                lin = self.w0 + np.sum(self.w[feats])
                # factorized interaction term using FM efficient formula
                sum_v = np.sum(self.V[feats, :], axis=0)  # shape (n_factors,)
                sum_v_sq = np.sum(self.V[feats, :] ** 2, axis=0)
                interaction = 0.5 * np.sum(sum_v ** 2 - sum_v_sq)
                pred = lin + interaction
                e = r - pred

                # update bias
                self.w0 += self.lr * (e - self.reg_w * self.w0)
                # update linear weights
                for idx in feats:
                    grad_w = -2.0 * e * 1.0 + 2.0 * self.reg_w * self.w[idx]
                    self.w[idx] -= self.lr * grad_w

                # update V for features
                # derivative for v_if: -2*e*(sum_v[f] - v_if) + 2*reg_v*v_if
                for f_idx in feats:
                    # vectorized across factors
                    v_if = self.V[f_idx, :]
                    grad_v = -2.0 * e * (sum_v - v_if) + 2.0 * self.reg_v * v_if
                    self.V[f_idx, :] -= self.lr * grad_v

        # clamp rating bounds not required here

    def predict(self, user: int, item: int) -> float:
        feats = self._make_feat_indices(user, item)
        if not feats:
            # fallback to item mean or global
            if self.item_means is not None and item in self.item_index:
                return float(self.item_means[self.item_index[item]])
            return 3.0
        lin = self.w0 + np.sum(self.w[feats])
        sum_v = np.sum(self.V[feats, :], axis=0)
        sum_v_sq = np.sum(self.V[feats, :] ** 2, axis=0)
        interaction = 0.5 * np.sum(sum_v ** 2 - sum_v_sq)
        pred = lin + interaction
        low, high = self.rating_bounds
        if pred < low: pred = low
        if pred > high: pred = high
        return float(pred)

    def recommend(self, user: int, k: int = 10) -> List[int]:
        if not self.item_index or not self.user_index:
            return []
        if user not in self.user_index:
            # popular items
            if self.item_means is None:
                return []
            top = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1][:k]]
            return top

        candidates = []
        rated = self._user_items.get(user, set())
        for it, idx in self.item_index.items():
            if it in rated:
                continue
            p = self.predict(user, it)
            candidates.append((it, p))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [int(it) for it, _ in candidates[:k]]
        # fill with popular if needed
        if len(top) < k and self.item_means is not None:
            popular = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1] if int(self.index_item[i]) not in top]
            for p in popular:
                top.append(p)
                if len(top) >= k:
                    break
        return top[:k]
