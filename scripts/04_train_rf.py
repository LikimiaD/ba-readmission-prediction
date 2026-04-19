import logging

import joblib

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import split_features_by_type, xy_split
from src.models import random_forest
from src.models.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion,
    plot_pr,
    plot_roc,
    split_metrics,
)
from src.utils.io import load_json, load_parquet, update_json

MODEL_NAME = 'random_forest'
log = logging.getLogger(MODEL_NAME)


def main():
    train = load_parquet(PROCESSED_DIR / 'train.parquet')
    val = load_parquet(PROCESSED_DIR / 'val.parquet')

    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    feature_types = split_features_by_type(train)

    pipeline = random_forest.build_pipeline(feature_types)
    pipeline, fit_seconds = random_forest.fit(pipeline, X_train, y_train)

    y_proba_val = random_forest.predict_proba(pipeline, X_val)
    m_val = compute_metrics(MODEL_NAME, 'val', y_val.to_numpy(), y_proba_val)

    plot_roc(y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'roc' / 'random_forest.png')
    plot_pr(y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'pr' / 'random_forest.png')
    plot_confusion(
        y_val.to_numpy(), y_proba_val, MODEL_NAME, FIGURES_DIR / 'confusion' / 'random_forest.png',
    )

    joblib.dump(pipeline, MODELS_DIR / 'random_forest.pkl', compress=3)

    best_threshold, _ = find_optimal_threshold(y_val.to_numpy(), y_proba_val)

    update_json(METRICS_PATH, {
        MODEL_NAME: {
            'val': split_metrics(m_val),
            'optimal_threshold': best_threshold,
            'fit_time_seconds': float(fit_seconds),
        }
    })

    if METRICS_PATH.exists():
        baseline_auc = (
            load_json(METRICS_PATH).get('baseline', {}).get('val', {}).get('roc_auc')
        )
        if baseline_auc is not None and m_val['roc_auc'] < baseline_auc:
            log.warning(
                'RF ROC-AUC %.3f < baseline %.3f — investigate data, not the model.',
                m_val['roc_auc'], baseline_auc,
            )

    print(
        f"\n[random_forest] val ROC-AUC={m_val['roc_auc']:.4f} "
        f"PR-AUC={m_val['pr_auc']:.4f} F1={m_val['f1']:.4f} Brier={m_val['brier']:.4f}"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
