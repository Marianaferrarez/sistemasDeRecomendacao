"""Bayesian Personalized Ranking (BPR) matrix factorization recommender.

Implements pairwise BPR optimization with SGD and negative sampling.
Treats interactions with rating >= `pos_threshold` as positive implicit feedback.
"""
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from core.recommender_registry import register_recommender


@register_recommender('bpr')
class BPRRecommender:
    def __init__(self, n_factors: int = 20, lr: float = 0.05, reg: float = 0.0025, n_epochs: int = 10, neg_samples: int = 1, pos_threshold: float = 4.0, verbose: bool = False):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.neg_samples = int(neg_samples)
        self.pos_threshold = float(pos_threshold)
        self.verbose = bool(verbose)

        self.user_index: Dict[int, int] = {}
        self.index_user: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.index_item: Dict[int, int] = {}

        self.P = None
        self.Q = None
        self.bi = None
        self.mu = 0.0

        self.user_pos: Dict[int, Set[int]] = {}
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

        rng = np.random.RandomState(123)
        self.P = 0.01 * rng.randn(n_users, self.n_factors)
        self.Q = 0.01 * rng.randn(n_items, self.n_factors)
        self.bi = np.zeros(n_items, dtype=float)

        # build positive interactions by threshold
        self.user_pos = {}
        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            it = int(row['item_id'])
            r = float(row['rating'])
            if r >= self.pos_threshold:
                self.user_pos.setdefault(u, set()).add(it)

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
        self.mu = float(np.mean(all_ratings)) if all_ratings.size > 0 else 0.0

        # prepare lists for sampling
        all_items = list(self.item_index.keys())
        users_with_pos = [u for u, s in self.user_pos.items() if len(s) > 0]

        if len(users_with_pos) == 0:
            return

        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'BPR epoch {epoch+1}/{self.n_epochs}')
            # iterate positives and sample negatives
            for u in users_with_pos:
                pos_items = list(self.user_pos[u])
                for i in pos_items:
                    for _ in range(self.neg_samples):
                        # sample negative
                        j = rng.choice(all_items)
                        while j in self.user_pos[u]:
                            j = rng.choice(all_items)
                        # map indices
                        uidx = self.user_index[u]
                        iidx = self.item_index[i]
                        jidx = self.item_index[j]

                        x_ui = self.P[uidx].dot(self.Q[iidx]) + self.bi[iidx]
                        x_uj = self.P[uidx].dot(self.Q[jidx]) + self.bi[jidx]
                        x_uij = x_ui - x_uj
                        # sigmoid(-x_uij)
                        sig = 1.0 / (1.0 + np.exp(x_uij))

                        # gradients
                        grad_p = sig * (self.Q[iidx] - self.Q[jidx]) - self.reg * self.P[uidx]
                        grad_qi = sig * self.P[uidx] - self.reg * self.Q[iidx]
                        grad_qj = -sig * self.P[uidx] - self.reg * self.Q[jidx]
                        grad_bi = sig - self.reg * self.bi[iidx]
                        grad_bj = -sig - self.reg * self.bi[jidx]

                        self.P[uidx] += self.lr * grad_p
                        self.Q[iidx] += self.lr * grad_qi
                        self.Q[jidx] += self.lr * grad_qj
                        self.bi[iidx] += self.lr * grad_bi
                        self.bi[jidx] += self.lr * grad_bj

    def predict(self, user: int, item: int) -> float:
        if self.P is None or self.Q is None:
            return 0.0
        uidx = self.user_index.get(user, None)
        iidx = self.item_index.get(item, None)
        if uidx is None and iidx is None:
            return 0.0
        if uidx is None:
            return float(self.item_means[iidx]) if iidx is not None and self.item_means is not None and self.item_means[iidx] > 0 else 0.0
        if iidx is None:
            return 0.0
        return float(self.P[uidx].dot(self.Q[iidx]) + self.bi[iidx])

    def recommend(self, user: int, k: int = 10) -> List[int]:
        if self.P is None or self.Q is None:
            return []
        uidx = self.user_index.get(user, None)
        n_items = self.Q.shape[0]
        if uidx is None:
            # popular
            if self.item_means is None:
                return []
            top = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1][:k]]
            return top

        scores = self.P[uidx].dot(self.Q.T) + self.bi
        # mask positives
        pos = self.user_pos.get(user, set())
        candidates = []
        for iidx in range(n_items):
            it = self.index_item[iidx]
            if it in pos:
                continue
            candidates.append((it, float(scores[iidx])))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [int(it) for it, _ in candidates[:k]]
        # fill if needed
        if len(top) < k and self.item_means is not None:
            popular = [int(self.index_item[i]) for i in np.argsort(self.item_means)[::-1] if int(self.index_item[i]) not in top]
            for p in popular:
                top.append(p)
                if len(top) >= k:
                    break
        return top[:k]
