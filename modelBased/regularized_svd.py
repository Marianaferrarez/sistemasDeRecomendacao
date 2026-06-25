"""Regularized SVD (matrix factorization) recommender.

Implements SGD-based matrix factorization with user/item biases and L2 regularization.
API matches other recommenders: `fit(train_df)`, `predict(user,item)`, `recommend(user,k)`.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('svd')
class RegularizedSVDRecommender:
    def __init__(self, n_factors: int = 20, lr: float = 0.005, reg: float = 0.02, n_epochs: int = 20, verbose: bool = False, rating_bounds=(1.0,5.0)):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.verbose = bool(verbose)
        self.rating_bounds = rating_bounds

        # mappings and trained params
        self.user_index: Dict[int, int] = {}
        self.index_user: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}

        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = 3.0

        # item means for fallback/popularity
        self.item_means = None

    def fit(self, train_df: pd.DataFrame):
        users = sorted(train_df['user_id'].unique())
        items = sorted(train_df['item_id'].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.index_user = {i: u for u, i in self.user_index.items()}
        self.item_index = {it: j for j, it in enumerate(items)}
        self.index_item = {j: it for it, j in self.item_index.items()}

        n_users = len(users)
        n_items = len(items)

        # initialize factors and biases
        rng = np.random.RandomState(42)
        self.P = 0.01 * rng.randn(n_users, self.n_factors)
        self.Q = 0.01 * rng.randn(n_items, self.n_factors)
        self.bu = np.zeros(n_users, dtype=float)
        self.bi = np.zeros(n_items, dtype=float)

        ratings = train_df['rating'].astype(float).values
        self.mu = float(np.mean(ratings)) if ratings.size > 0 else 3.0

        # precompute item means
        item_sums = np.zeros(n_items, dtype=float)
        item_counts = np.zeros(n_items, dtype=int)
        for _, row in train_df.iterrows():
            it = int(row['item_id'])
            idx = self.item_index[it]
            item_sums[idx] += float(row['rating'])
            item_counts[idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.item_means = np.where(item_counts > 0, item_sums / item_counts, 0.0)

        self._user_ratings = {}
        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            it = int(row['item_id'])
            self._user_ratings.setdefault(u, {})[it] = float(row['rating'])

        rows = list(train_df.itertuples(index=False))
        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'SVD epoch {epoch+1}/{self.n_epochs}')
            rng.shuffle(rows)
            for r in rows:
                # r tuple fields: (user_id, item_id, rating) or depending on DataFrame order
                # support both namedtuple and simple tuple
                try:
                    u = int(r.user_id)
                    it = int(r.item_id)
                    rating = float(r.rating)
                except AttributeError:
                    u = int(r[0]); it = int(r[1]); rating = float(r[2])

                uidx = self.user_index[u]
                iidx = self.item_index[it]
                pred = self.mu + self.bu[uidx] + self.bi[iidx] + self.P[uidx].dot(self.Q[iidx])
                e = rating - pred

                # update biases
                self.bu[uidx] += self.lr * (e - self.reg * self.bu[uidx])
                self.bi[iidx] += self.lr * (e - self.reg * self.bi[iidx])
                # update latent factors
                pu = self.P[uidx]
                qi = self.Q[iidx]
                self.P[uidx] += self.lr * (e * qi - self.reg * pu)
                self.Q[iidx] += self.lr * (e * pu - self.reg * qi)

        # clamp biases/factors if necessary (not required)

    def predict(self, user: int, item: int) -> float:
        if self.P is None or self.Q is None:
            return self.mu

        uidx = self.user_index.get(user, None)
        iidx = self.item_index.get(item, None)
        if uidx is None and iidx is None:
            return self.mu
        if uidx is None:
            # unknown user: use item mean or global
            return float(self.item_means[iidx]) if iidx is not None and self.item_means is not None and self.item_means[iidx] > 0 else self.mu
        if iidx is None:
            return float(self.mu + self.bu[uidx])

        pred = self.mu + self.bu[uidx] + self.bi[iidx] + self.P[uidx].dot(self.Q[iidx])
        low, high = self.rating_bounds
        if pred < low: pred = low
        if pred > high: pred = high
        return float(pred)

    def recommend(self, user: int, k: int = 10) -> List[int]:
        if self.P is None or self.Q is None:
            return []
        uidx = self.user_index.get(user, None)
        n_items = self.Q.shape[0]
        if uidx is None:
            # recommend popular items
            if self.item_means is None:
                return []
            top = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1][:k]]
            return top

        # compute predictions for all items
        preds = self.mu + self.bu[uidx] + self.bi + self.P[uidx].dot(self.Q.T)

        # mask items the user rated (if we had train interactions cached, but simplest: assume user rated items are in user_index mapping?)
        # To detect rated items, we need training info; try to approximate by comparing to item_means presence.
        # Safer: collect items user rated from training by adding cache in fit. Implement simple cache.
        try:
            rated_items = set(self._user_ratings.get(int(user), {}).keys())
        except Exception:
            rated_items = set()

        candidates = []
        for iidx in range(n_items):
            it = self.index_item[iidx]
            if it in rated_items:
                continue
            candidates.append((it, float(preds[iidx])))

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
