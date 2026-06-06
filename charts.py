"""Generate charts (scatter, ROC, confusion matrix) from results CSVs and save as PDFs.

Usage:
    python charts.py              # process all algos found in `results/` (predictions_*.csv)
    python charts.py simple_memory --threshold 4.0

Outputs saved under `results/charts/` as PDF files.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, mean_absolute_error


def find_algorithms_from_results(results_dir: str):
    pattern = os.path.join(results_dir, 'predictions_*.csv')
    files = glob.glob(pattern)
    algos = [os.path.splitext(os.path.basename(f))[0].replace('predictions_', '') for f in files]
    return sorted(algos)


def load_predictions(results_dir: str, algo: str):
    path = os.path.join(results_dir, f'predictions_{algo}.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def plot_scatter(df: pd.DataFrame, outpath: str):
    if df is None or df.empty:
        return False
    y = df['rating_true'].to_numpy()
    yhat = df['rating_pred'].to_numpy()
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = mean_absolute_error(y, yhat)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, alpha=0.4, s=10)
    mn = min(np.min(y), np.min(yhat))
    mx = max(np.max(y), np.max(yhat))
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
    plt.xlabel('True rating')
    plt.ylabel('Predicted rating')
    plt.title(f'Scatter: RMSE={rmse:.3f}, MAE={mae:.3f}')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def plot_roc(df: pd.DataFrame, outpath: str, threshold: float = 4.0):
    if df is None or df.empty:
        return False
    y_true = (df['rating_true'] >= threshold).astype(int).to_numpy()
    scores = df['rating_pred'].to_numpy()
    if len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC (threshold={threshold})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def plot_confusion(df: pd.DataFrame, outpath: str, threshold: float = 4.0, pred_threshold: float = 4.0):
    if df is None or df.empty:
        return False
    y_true = (df['rating_true'] >= threshold).astype(int).to_numpy()
    y_pred = (df['rating_pred'] >= pred_threshold).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion (thr_true={threshold}, thr_pred={pred_threshold})')
    plt.colorbar(im)
    classes = ['neg', 'pos']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('algos', nargs='*', help='Algorithm names to plot (registered names). If omitted, uses results/*.csv')
    p.add_argument('--results-dir', default='results', help='Results folder')
    p.add_argument('--out-dir', default=None, help='Directory to save charts (default: results/charts)')
    p.add_argument('--threshold', type=float, default=4.0, help='Ground-truth rating threshold for relevance')
    p.add_argument('--pred-threshold', type=float, default=4.0, help='Predicted rating threshold for confusion matrix')
    args = p.parse_args()

    results_dir = args.results_dir
    if args.out_dir:
        out_base = args.out_dir
    else:
        out_base = os.path.join(results_dir, 'charts')
    ensure_dir(out_base)

    if args.algos:
        algos = args.algos
    else:
        algos = find_algorithms_from_results(results_dir)

    if not algos:
        print('No algorithms found in results folder:', results_dir)
        return

    for algo in algos:
        print('Processing', algo)
        df = load_predictions(results_dir, algo)
        algo_dir = os.path.join(out_base, algo)
        ensure_dir(algo_dir)
        scatter_path = os.path.join(algo_dir, f'{algo}_scatter.pdf')
        roc_path = os.path.join(algo_dir, f'{algo}_roc.pdf')
        cm_path = os.path.join(algo_dir, f'{algo}_confusion.pdf')

        ok1 = plot_scatter(df, scatter_path)
        ok2 = plot_roc(df, roc_path, threshold=args.threshold)
        ok3 = plot_confusion(df, cm_path, threshold=args.threshold, pred_threshold=args.pred_threshold)

        if not any((ok1, ok2, ok3)):
            print(f'  no valid prediction data for {algo}, skipped')
        else:
            print(f'  charts saved to {algo_dir}')


if __name__ == '__main__':
    main()
