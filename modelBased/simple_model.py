"""A minimal model-based recommender stub.

This is a lightweight placeholder that uses a simple global bias model
(`rating ≈ global + user_bias + item_bias`) fitted by least squares.
It's intended as a starting point for swapping in matrix factorization or
other model-based approaches later.
"""
import numpy as np
from core.recommender_registry import register_recommender


@register_recommender('simple_model')
class SimpleModelRecommender:
    def __init__(self, train_df=None, reg=1e-5):
        self.user_bias = {}
        self.item_bias = {}
        self.global_mean = 3.0
        self.reg = reg
        if train_df is not None:
            self.fit(train_df)

    def fit(self, train_df):
        ratings = train_df['rating'].astype(float).tolist()
        self.global_mean = float(np.mean(ratings)) if ratings else 3.0
        users = {}
        items = {}
        user_counts = {}
        item_counts = {}
        self._user_items = {}
        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            r = float(row['rating'])
            users.setdefault(u, 0.0)
            users[u] += (r - self.global_mean)
            user_counts[u] = user_counts.get(u, 0) + 1
            items.setdefault(i, 0.0)
            items[i] += (r - self.global_mean)
            item_counts[i] = item_counts.get(i, 0) + 1
            self._user_items.setdefault(u, set()).add(i)
        self.user_bias = {u: users[u] / (user_counts[u] + self.reg) for u in users}
        self.item_bias = {i: items[i] / (item_counts[i] + self.reg) for i in items}

    def predict(self, user, item):
        ub = self.user_bias.get(user, 0.0)
        ib = self.item_bias.get(item, 0.0)
        return float(self.global_mean + ub + ib)

    def recommend(self, user, k=10):
        rated = self._user_items.get(user, set())
        items_sorted = sorted(self.item_bias.items(), key=lambda x: x[1], reverse=True)
        return [int(i) for i, _ in items_sorted if i not in rated][:k]
