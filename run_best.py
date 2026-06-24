"""Re-run the full evaluation pipeline using the best hyperparameters
found during tuning. Overwrites results/ with optimized outputs."""

import os
import sys
from core.evaluation import StudyRunner, load_ml100k_ratings

from memoryBased.simple_memory import SimpleMemoryRecommender
from memoryBased.user_based import UserBasedRecommender
from memoryBased.item_based import ItemBasedRecommender
from memoryBased.slope_one import SlopeOneRecommender
from modelBased.simple_model import SimpleModelRecommender
from modelBased.regularized_svd import RegularizedSVDRecommender
from modelBased.bpr import BPRRecommender
from modelBased.fm import FMRecommender


BEST_PARAMS = {
    "simple_memory": {"cls": SimpleMemoryRecommender, "params": {"alpha": 0.4}},
    "user_based":    {"cls": UserBasedRecommender,    "params": {"k": 30}},
    "item_based":    {"cls": ItemBasedRecommender,    "params": {"k": 30}},
    "slope_one":     {"cls": SlopeOneRecommender,     "params": {"shrink": 100.0}},
    "simple_model":  {"cls": SimpleModelRecommender,  "params": {"reg": 10.0}},
    "svd":           {"cls": RegularizedSVDRecommender,"params": {"n_factors": 50, "lr": 0.01, "reg": 0.01}},
    "bpr":           {"cls": BPRRecommender,          "params": {"n_factors": 50, "lr": 0.05, "reg": 0.0025}},
    "fm":            {"cls": FMRecommender,           "params": {"n_factors": 20, "lr": 0.01, "reg_w": 0.001, "reg_v": 0.001}},
}


def main():
    data_dir = os.path.join("movielens", "ml-100k")
    print("Loading data...")
    train = load_ml100k_ratings(data_dir, "ua.base")
    test = load_ml100k_ratings(data_dir, "ua.test")

    chosen = BEST_PARAMS
    if len(sys.argv) > 1:
        requested = [a.lower() for a in sys.argv[1:]]
        chosen = {k: v for k, v in BEST_PARAMS.items() if k in requested}

    algos = {}
    for name, spec in chosen.items():
        print(f"Training {name} with {spec['params']}...")
        alg = spec["cls"](**spec["params"])
        alg.fit(train)
        algos[name] = alg

    runner = StudyRunner()
    save_dir = "results"
    results = runner.run(algos, train, test, k=10, save_dir=save_dir)

    print("\nEvaluation results (best hyperparameters):")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")


if __name__ == "__main__":
    main()
