from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURE_DPI

_SPLIT_KEYS = ('roc_auc', 'pr_auc', 'f1', 'brier', 'n_samples')


def compute_metrics(model_name, split, y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'model_name': model_name,
        'split': split,
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'f1': float(f1_score(y_true, y_pred)),
        'brier': float(brier_score_loss(y_true, y_proba)),
        'n_samples': int(len(y_true)),
    }


def split_metrics(m):
    return {k: m[k] for k in _SPLIT_KEYS}


def find_optimal_threshold(y_true, y_proba, metric: str = 'f1') -> tuple[float, float]:
    if metric != 'f1':
        raise NotImplementedError(f"Only metric='f1' is supported, got {metric!r}")
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def plot_roc(y_true, y_proba, model_name, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = float(roc_auc_score(y_true, y_proba))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC — {model_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    _finalize(fig, out_path)
    return out_path


def plot_pr(y_true, y_proba, model_name, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = float(average_precision_score(y_true, y_proba))
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    ax.hlines(baseline, 0, 1, linestyles='--', colors='grey', label=f'Base rate = {baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall — {model_name}')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    _finalize(fig, out_path)
    return out_path


def plot_confusion(y_true, y_proba, model_name, out_path, threshold: float = 0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['no_30d', 'readmit_30d'])
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f'Confusion matrix — {model_name} @ thr={threshold:.2f}')
    _finalize(fig, out_path)
    return out_path


def plot_calibration(y_true, y_proba, model_name, out_path, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='quantile')
    brier = float(brier_score_loss(y_true, y_proba))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True,
    )
    ax_top.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfectly calibrated')
    ax_top.plot(mean_pred, frac_pos, marker='o', label=f'{model_name} (Brier = {brier:.4f})')
    ax_top.set_ylabel('Fraction of positives')
    ax_top.set_title(f'Reliability diagram — {model_name}')
    ax_top.legend(loc='upper left')
    ax_top.grid(alpha=0.3)

    ax_bot.hist(y_proba, bins=20, color='steelblue', edgecolor='white')
    ax_bot.set_xlabel('Predicted probability')
    ax_bot.set_ylabel('Count')
    ax_bot.grid(alpha=0.3)

    _finalize(fig, out_path)
    return out_path


def plot_roc_overlay(results, out_path, title: str = 'ROC — all models'):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = float(roc_auc_score(y_true, y_proba))
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    _finalize(fig, out_path)
    return out_path


def plot_pr_overlay(results, out_path, title: str = 'Precision-Recall — all models'):
    fig, ax = plt.subplots(figsize=(6, 5))
    first_name = next(iter(results))
    base_rate = float(np.mean(results[first_name][0]))
    ax.hlines(base_rate, 0, 1, linestyles='--', colors='grey', label=f'Base rate = {base_rate:.3f}')
    for name, (y_true, y_proba) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = float(average_precision_score(y_true, y_proba))
        ax.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    _finalize(fig, out_path)
    return out_path


def _finalize(fig, path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor='white')
    plt.close(fig)
