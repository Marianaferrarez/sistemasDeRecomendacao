"""Sensitivity analysis: varies one hyperparameter per algorithm,
evaluates RMSE via 5-fold CV, saves results CSV and generates plots.

Usage:
    python sensitivity_analysis.py              # all algorithms
    python sensitivity_analysis.py svd fm       # specific algorithms
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

from core.evaluation import load_ml100k_ratings

DATA_DIR = os.path.join("movielens", "ml-100k")
N_FOLDS = 5
OUT_DIR = os.path.join("results", "sensitivity")


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


def run_sensitivity(cls, param_name, param_values, fixed_params, folds):
    results = []
    for val in param_values:
        params = {**fixed_params, param_name: val}
        fold_rmses = []
        for train, test in folds:
            alg = cls(**params)
            alg.fit(train)
            rmse = evaluate_rmse(alg, test)
            fold_rmses.append(rmse)
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        results.append({"value": val, "mean_rmse": mean_rmse, "std_rmse": std_rmse})
        print(f"    {param_name}={val}  =>  RMSE = {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    return results


def plot_sensitivity(algo_name, param_name, results, out_dir):
    values = [r["value"] for r in results]
    means = [r["mean_rmse"] for r in results]
    stds = [r["std_rmse"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(values, means, yerr=stds, marker='o', capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("RMSE (5-fold CV)", fontsize=12)
    ax.set_title(f"Sensitivity: {algo_name} — {param_name}", fontsize=13)
    ax.grid(True, alpha=0.3)

    best_idx = np.argmin(means)
    ax.annotate(f"best={values[best_idx]}",
                xy=(values[best_idx], means[best_idx]),
                xytext=(10, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{algo_name}_{param_name}.pdf")
    plt.savefig(path)
    plt.close()
    return path


def main():
    from memoryBased.simple_memory import SimpleMemoryRecommender
    from memoryBased.user_based import UserBasedRecommender
    from memoryBased.item_based import ItemBasedRecommender
    from memoryBased.slope_one import SlopeOneRecommender
    from modelBased.simple_model import SimpleModelRecommender
    from modelBased.regularized_svd import RegularizedSVDRecommender
    from modelBased.bpr import BPRRecommender
    from modelBased.fm import FMRecommender

    analyses = {
        "simple_memory": {
            "cls": SimpleMemoryRecommender,
            "param_name": "alpha",
            "param_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "fixed_params": {},
        },
        "user_based": {
            "cls": UserBasedRecommender,
            "param_name": "k",
            "param_values": [5, 10, 15, 20, 30, 40, 50, 80],
            "fixed_params": {},
        },
        "item_based": {
            "cls": ItemBasedRecommender,
            "param_name": "k",
            "param_values": [5, 10, 15, 20, 30, 40, 50, 80],
            "fixed_params": {},
        },
        "slope_one": {
            "cls": SlopeOneRecommender,
            "param_name": "shrink",
            "param_values": [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0],
            "fixed_params": {},
        },
        "simple_model": {
            "cls": SimpleModelRecommender,
            "param_name": "reg",
            "param_values": [1e-7, 1e-5, 1e-3, 1e-1, 1.0, 5.0, 10.0, 25.0],
            "fixed_params": {},
        },
        "svd": {
            "cls": RegularizedSVDRecommender,
            "param_name": "n_factors",
            "param_values": [5, 10, 20, 30, 50, 75, 100],
            "fixed_params": {"lr": 0.01, "reg": 0.01},
        },
        "bpr": {
            "cls": BPRRecommender,
            "param_name": "n_factors",
            "param_values": [5, 10, 20, 30, 50, 75, 100],
            "fixed_params": {"lr": 0.05, "reg": 0.0025},
        },
        "fm": {
            "cls": FMRecommender,
            "param_name": "n_factors",
            "param_values": [5, 10, 15, 20, 30, 50],
            "fixed_params": {"lr": 0.01, "reg_w": 0.001, "reg_v": 0.001},
        },
    }

    algos_filter = None
    if len(sys.argv) > 1:
        algos_filter = [a.lower() for a in sys.argv[1:]]
        analyses = {k: v for k, v in analyses.items() if k in algos_filter}

    print("Loading folds...")
    folds = load_folds()
    print(f"Loaded {N_FOLDS} folds.\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for name, spec in analyses.items():
        print(f"=== Sensitivity: {name} ({spec['param_name']}) ===")
        t0 = time.time()
        results = run_sensitivity(
            spec["cls"], spec["param_name"], spec["param_values"],
            spec["fixed_params"], folds
        )
        elapsed = time.time() - t0

        path = plot_sensitivity(name, spec["param_name"], results, OUT_DIR)
        print(f"  Chart saved to {path}")
        print(f"  Time: {elapsed:.1f}s\n")

        for r in results:
            all_rows.append({
                "algorithm": name,
                "param_name": spec["param_name"],
                "param_value": r["value"],
                "mean_rmse": r["mean_rmse"],
                "std_rmse": r["std_rmse"],
            })

    new_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "sensitivity_results.csv")
    if os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["algorithm"].isin(new_df["algorithm"].unique())]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    print(f"All results saved to {csv_path}")


if __name__ == "__main__":
    main()
