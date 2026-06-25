from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def load_ml100k_ratings(path: str, filename: str = 'u.data') -> pd.DataFrame:
    full = path.rstrip('/') + '/' + filename
    try:
        df = pd.read_csv(full, sep="\t", header=None, engine='python')
    except Exception:
        df = pd.read_csv(full, sep=r'\s+', header=None, engine='python')
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    else:
        raise ValueError(f'unexpected rating file format: {full}')
    return df


class StudyRunner:
    def __init__(self):
        pass

    @staticmethod
    def _group_truth_by_user(df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
        grouped = {}
        for u, g in df.groupby('user_id'):
            grouped[int(u)] = {int(row['item_id']): float(row['rating']) for _, row in g.iterrows()}
        return grouped

    def evaluate_ratings(self, true_ratings: Iterable[float], pred_ratings: Iterable[float]) -> Dict[str, float]:
        y_true = np.array(list(true_ratings), dtype=float)
        y_pred = np.array(list(pred_ratings), dtype=float)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae}

    @staticmethod
    def _precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
        if not recommended:
            return 0.0
        recommended_k = recommended[:k]
        hits = sum(1 for i in recommended_k if i in relevant)
        return hits / float(k)

    @staticmethod
    def _recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
        if not relevant:
            return 0.0
        recommended_k = recommended[:k]
        hits = sum(1 for i in recommended_k if i in relevant)
        return hits / float(len(relevant))

    @staticmethod
    def _ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
        dcg = 0.0
        for idx, item in enumerate(recommended[:k], start=1):
            rel = 1.0 if item in relevant else 0.0
            dcg += rel / np.log2(idx + 1)
        ideal_rels = [1.0] * min(len(relevant), k)
        idcg = 0.0
        for idx, rel in enumerate(ideal_rels, start=1):
            idcg += rel / np.log2(idx + 1)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _apk(recommended: List[int], relevant: set, k: int) -> float:
        if not recommended:
            return 0.0
        score = 0.0
        hits = 0.0
        for i, p in enumerate(recommended[:k], start=1):
            if p in relevant and p not in recommended[:i-1]:
                hits += 1.0
                score += hits / i
        return score / min(len(relevant), k) if len(relevant) > 0 else 0.0

    def evaluate_recommendations(self, recommended_by_user: Dict[int, List[int]], true_by_user: Dict[int, Dict[int, float]], k: int = 10) -> Dict[str, float]:
        precision_list = []
        recall_list = []
        ndcg_list = []
        apk_list = []
        for user, recs in recommended_by_user.items():
            true_items = true_by_user.get(user, {})
            # binary relevance: threshold 4.0
            relevant = set([i for i, r in true_items.items() if r >= 4.0])
            precision_list.append(self._precision_at_k(recs, relevant, k))
            recall_list.append(self._recall_at_k(recs, relevant, k))
            ndcg_list.append(self._ndcg_at_k(recs, relevant, k))
            apk_list.append(self._apk(recs, relevant, k))
        return {
            f'Precision@{k}': float(np.mean(precision_list)) if precision_list else 0.0,
            f'Recall@{k}': float(np.mean(recall_list)) if recall_list else 0.0,
            f'NDCG@{k}': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            f'MAP@{k}': float(np.mean(apk_list)) if apk_list else 0.0,
        }

    def run(self, algorithms: Dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10, save_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        # prepare truth structures
        true_by_user = self._group_truth_by_user(test_df)
        results = {}
        # containers for detailed outputs
        predictions_details: Dict[str, List[Dict]] = {name: [] for name in algorithms}
        recommendations_details: Dict[str, List[Dict]] = {name: [] for name in algorithms}

        # rating predictions
        for name, alg in algorithms.items():
            # predict ratings for each test pair if predict method exists
            preds = []
            trues = []
            if hasattr(alg, 'predict'):
                for _, row in test_df.iterrows():
                    u = int(row['user_id'])
                    i = int(row['item_id'])
                    y = float(row['rating'])
                    try:
                        p = float(alg.predict(u, i))
                    except Exception:
                        p = np.nan
                    if not np.isnan(p):
                        preds.append(p)
                        trues.append(y)
                        predictions_details[name].append({'user_id': u, 'item_id': i, 'rating_true': y, 'rating_pred': p})
            rating_metrics = self.evaluate_ratings(trues, preds) if preds else {}

            # recommendation metrics
            recommended_by_user = {}
            if hasattr(alg, 'recommend'):
                users = list(test_df['user_id'].unique())
                for u in users:
                    try:
                        recs = list(alg.recommend(int(u), k))
                        recommended_by_user[int(u)] = recs
                    except Exception:
                        recs = []
                        recommended_by_user[int(u)] = []
                    # record recommendation details with hit flag
                    true_items = true_by_user.get(int(u), {})
                    relevant = set([i for i, r in true_items.items() if r >= 4.0])
                    for rank, item in enumerate(recs[:k], start=1):
                        hit = 1 if item in relevant else 0
                        recommendations_details[name].append({'user_id': int(u), 'rank': rank, 'item_id': int(item), 'hit': hit})
                rec_metrics = self.evaluate_recommendations(recommended_by_user, true_by_user, k=k)
            else:
                rec_metrics = {}

            results[name] = {**rating_metrics, **rec_metrics}

        # optionally save detailed outputs
        if save_dir:
            try:
                self._save_results(results, predictions_details, recommendations_details, save_dir)
            except Exception:
                pass
        return results

        # unreachable: kept for clarity

    def _save_results(self, results: Dict[str, Dict[str, float]], predictions_details: Dict[str, List[Dict]], recommendations_details: Dict[str, List[Dict]], save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        # save summary
        rows = []
        for name, metrics in results.items():
            row = {'algorithm': name}
            row.update(metrics)
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(save_dir, 'summary.csv'), index=False)

        # save detailed predictions and recommendations per algorithm
        for name, recs in predictions_details.items():
            dfp = pd.DataFrame(recs)
            if dfp.empty:
                # ensure columns exist
                dfp = pd.DataFrame(columns=['user_id', 'item_id', 'rating_true', 'rating_pred'])
            dfp.to_csv(os.path.join(save_dir, f'predictions_{name}.csv'), index=False)
        for name, recs in recommendations_details.items():
            dfr = pd.DataFrame(recs)
            if dfr.empty:
                dfr = pd.DataFrame(columns=['user_id', 'rank', 'item_id', 'hit'])
            dfr.to_csv(os.path.join(save_dir, f'recommendations_{name}.csv'), index=False)