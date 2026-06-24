"""Hyperparameter tuning via grid search with 5-fold cross-validation.

Uses the pre-built MovieLens 100k splits (u1..u5) as folds.
For each algorithm, varies at least one key hyperparameter, evaluates
RMSE on each fold, and reports mean +/- std across folds.

Usage:
    python hyperparameter_tuning.py              # tune all algorithms
    python hyperparameter_tuning.py user_based   # tune only user_based
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

from core.evaluation import load_ml100k_ratings


DATA_DIR = os.path.join("movielens", "ml-100k")
N_FOLDS = 5
RESULTS_DIR = "results"


def load_folds():
    folds = []
    for i in range(1, N_FOLDS + 1):
        train = load_ml100k_ratings(DATA_DIR, f"u{i}.base")
        test = load_ml100k_ratings(DATA_DIR, f"u{i}.test")
        folds.append((train, test))
    return folds


def evaluate_rmse(alg, test_df):
    users = test_df["user_id"].values
    items = test_df["item_id"].values
    ratings = test_df["rating"].values.astype(float)
    trues, preds = [], []
    for u, i, r in zip(users, items, ratings):
        try:
            p = float(alg.predict(int(u), int(i)))
        except Exception:
            continue
        if not np.isnan(p):
            trues.append(r)
            preds.append(p)
    if not trues:
        return float("nan")
    return sqrt(mean_squared_error(trues, preds))


def cross_validate(cls, param_grid, folds):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_rmses = []
        for fold_idx, (train, test) in enumerate(folds):
            alg = cls(**params)
            alg.fit(train)
            rmse = evaluate_rmse(alg, test)
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        results.append({**params, "mean_rmse": mean_rmse, "std_rmse": std_rmse})
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"    {param_str}  =>  RMSE = {mean_rmse:.4f} (+/- {std_rmse:.4f})")

    return results


def tune_all(algos_filter=None):
    from memoryBased.simple_memory import SimpleMemoryRecommender
    from memoryBased.user_based import UserBasedRecommender
    from memoryBased.item_based import ItemBasedRecommender
    from memoryBased.slope_one import SlopeOneRecommender
    from modelBased.simple_model import SimpleModelRecommender
    from modelBased.regularized_svd import RegularizedSVDRecommender
    from modelBased.bpr import BPRRecommender
    from modelBased.fm import FMRecommender

    algorithms = {
        "simple_memory": {
            "cls": SimpleMemoryRecommender,
            "grid": {
                "alpha": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            },
        },
        "slope_one": {
            "cls": SlopeOneRecommender,
            "grid": {
                "shrink": [0.0, 5.0, 10.0, 25.0, 50.0, 100.0],
            },
        },
        "simple_model": {
            "cls": SimpleModelRecommender,
            "grid": {
                "reg": [1e-7, 1e-5, 1e-3, 1e-1, 1.0, 10.0],
            },
        },
        "svd": {
            "cls": RegularizedSVDRecommender,
            "grid": {
                "n_factors": [10, 20, 50],
                "lr": [0.005, 0.01],
                "reg": [0.01, 0.02, 0.05],
            },
        },
        "bpr": {
            "cls": BPRRecommender,
            "grid": {
                "n_factors": [10, 20, 50],
                "lr": [0.01, 0.05],
                "reg": [0.0025, 0.01],
            },
        },
        "fm": {
            "cls": FMRecommender,
            "grid": {
                "n_factors": [5, 10, 20],
                "lr": [0.005, 0.01],
                "reg_w": [0.001, 0.01],
                "reg_v": [0.001, 0.01],
            },
        },
        "user_based": {
            "cls": UserBasedRecommender,
            "grid": {
                "k": [10, 20, 30, 50],
            },
        },
        "item_based": {
            "cls": ItemBasedRecommender,
            "grid": {
                "k": [10, 20, 30, 50],
            },
        },
    }

    if algos_filter:
        algorithms = {k: v for k, v in algorithms.items() if k in algos_filter}

    print("Loading folds...")
    folds = load_folds()
    print(f"Loaded {N_FOLDS} folds.\n")

    all_results = {}
    best_params = {}

    for name, spec in algorithms.items():
        print(f"=== Tuning: {name} ===")
        n_combos = 1
        for vals in spec["grid"].values():
            n_combos *= len(vals)
        print(f"  Grid size: {n_combos} combinations x {N_FOLDS} folds\n")

        t0 = time.time()
        results = cross_validate(spec["cls"], spec["grid"], folds)
        elapsed = time.time() - t0

        results_sorted = sorted(results, key=lambda x: x["mean_rmse"])
        best = results_sorted[0]
        best_param_dict = {k: v for k, v in best.items() if k not in ("mean_rmse", "std_rmse")}
        best_params[name] = best_param_dict

        print(f"\n  Best: {best_param_dict}  =>  RMSE = {best['mean_rmse']:.4f} (+/- {best['std_rmse']:.4f})")
        print(f"  Time: {elapsed:.1f}s\n")

        all_results[name] = results_sorted

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for name, results in all_results.items():
        for r in results:
            row = {"algorithm": name}
            row.update(r)
            rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "hyperparameter_tuning.csv")
    df.to_csv(out_path, index=False)
    print(f"Full results saved to {out_path}")

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS SUMMARY")
    print("=" * 60)
    summary_rows = []
    for name, params in best_params.items():
        best_result = all_results[name][0]
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {name:20s}  {param_str:40s}  RMSE={best_result['mean_rmse']:.4f}")
        summary_rows.append({
            "algorithm": name,
            "best_params": param_str,
            "mean_rmse": best_result["mean_rmse"],
            "std_rmse": best_result["std_rmse"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "best_hyperparameters.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nBest params saved to {summary_path}")

    return best_params


if __name__ == "__main__":
    algos_filter = None
    if len(sys.argv) > 1:
        algos_filter = [a.lower() for a in sys.argv[1:]]
    tune_all(algos_filter)
