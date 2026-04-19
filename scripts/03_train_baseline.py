import logging

import joblib

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import split_features_by_type, xy_split
from src.models import baseline
from src.models.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion,
    plot_pr,
    plot_roc,
    split_metrics,
)
from src.utils.io import load_parquet, update_json

MODEL_NAME = 'baseline'
log = logging.getLogger(MODEL_NAME)


def main():
    train = load_parquet(PROCESSED_DIR / 'train.parquet')
    val = load_parquet(PROCESSED_DIR / 'val.parquet')

    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    feature_types = split_features_by_type(train)

    pipeline = baseline.build_pipeline(feature_types)
    pipeline, fit_seconds = baseline.fit(pipeline, X_train, y_train)

    y_proba_val = baseline.predict_proba(pipeline, X_val)
    m_val = compute_metrics(MODEL_NAME, 'val', y_val.to_numpy(), y_proba_val)

    plot_roc(y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'roc' / 'baseline.png')
    plot_pr(y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'pr' / 'baseline.png')
    plot_confusion(
        y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'confusion' / 'baseline.png',
    )

    joblib.dump(pipeline, MODELS_DIR / 'baseline.pkl')

    best_threshold, _ = find_optimal_threshold(y_val.to_numpy(), y_proba_val)

    update_json(METRICS_PATH, {
        MODEL_NAME: {
            'val': split_metrics(m_val),
            'optimal_threshold': best_threshold,
            'fit_time_seconds': float(fit_seconds),
        }
    })
    if m_val['roc_auc'] < 0.60:
        log.warning('Baseline ROC-AUC %.3f < 0.60 — possible leakage or bug.', m_val['roc_auc'])

    print(
        f"\n[baseline] val ROC-AUC={m_val['roc_auc']:.4f} "
        f"PR-AUC={m_val['pr_auc']:.4f} F1={m_val['f1']:.4f} Brier={m_val['brier']:.4f}"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
