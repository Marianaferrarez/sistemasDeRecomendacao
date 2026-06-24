"""A simple memory-based recommender stub.

This implementation is intentionally minimal: it computes item popularity and
user/item means to provide predictable predictions and a simple popularity-
based recommendation list. It's a starting point to add proper kNN or
similarity-based methods later.
"""
from collections import Counter, defaultdict
import numpy as np
from core.recommender_registry import register_recommender


@register_recommender('simple_memory')
class SimpleMemoryRecommender:
    def __init__(self, train_df=None, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.user_means = {}
        self.item_means = {}
        self.global_mean = 3.0
        self.popular_items = []
        if train_df is not None:
            self.fit(train_df)

    def fit(self, train_df):
        users = defaultdict(list)
        items = defaultdict(list)
        for _, row in train_df.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            r = float(row['rating'])
            users[u].append(r)
            items[i].append(r)
        self.user_means = {u: np.mean(rs) for u, rs in users.items()}
        self.item_means = {i: np.mean(rs) for i, rs in items.items()}
        self.global_mean = np.mean([r for rs in users.values() for r in rs]) if users else 3.0
        counts = Counter([int(row['item_id']) for _, row in train_df.iterrows()])
        self.popular_items = [i for i, _ in counts.most_common()]

    def predict(self, user, item):
        um = self.user_means.get(user, None)
        im = self.item_means.get(item, None)
        if um is None and im is None:
            return self.global_mean
        if um is None:
            return im
        if im is None:
            return um
        return self.alpha * um + (1 - self.alpha) * im

    def recommend(self, user, k=10):
        return self.popular_items[:k]
