import logging

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import split_features_by_type, xy_split
from src.models import catboost_model
from src.models.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion,
    plot_pr,
    plot_roc,
    split_metrics,
)
from src.utils.io import load_parquet, update_json

MODEL_NAME = 'catboost'
N_TRIALS = 150
TIMEOUT_SECONDS = 1800
log = logging.getLogger(MODEL_NAME)


def main():
    train = load_parquet(PROCESSED_DIR / 'train.parquet')
    val = load_parquet(PROCESSED_DIR / 'val.parquet')

    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    feature_types = split_features_by_type(train)

    study = catboost_model.tune(
        X_train, y_train, X_val, y_val, feature_types,
        n_trials=N_TRIALS, timeout_seconds=TIMEOUT_SECONDS,
    )
    best_params = dict(study.best_params)

    y_proba_val_honest, honest_best_iteration = catboost_model.honest_val_predictions(
        best_params, X_train, y_train, X_val, y_val, feature_types
    )
    m_val = compute_metrics(MODEL_NAME, 'val', y_val.to_numpy(), y_proba_val_honest)
    log.info('Honest val: %s (early-stopped iter=%d)', m_val, honest_best_iteration)

    plot_roc(y_val.to_numpy(), y_proba_val_honest, MODEL_NAME, FIGURES_DIR / 'roc' / 'catboost.png')
    plot_pr(y_val.to_numpy(), y_proba_val_honest, MODEL_NAME, FIGURES_DIR / 'pr' / 'catboost.png')
    plot_confusion(
        y_val.to_numpy(), y_proba_val_honest, MODEL_NAME,
        FIGURES_DIR / 'confusion' / 'catboost.png',
    )

    # Финальный фит на train+val — из него считаем test и SHAP. Капим
    # iterations на best_iteration × scale из честного train-only фита,
    # иначе рефит перетренировывается (Red Flag §7.2 TASK-2b).
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    iterations_cap = None
    if honest_best_iteration > 0:
        scale = 1.0 + (len(X_val) / max(len(X_train), 1))
        iterations_cap = max(int(honest_best_iteration * scale), 1)
    model, fit_seconds = catboost_model.fit_final(
        best_params, X_full, y_full, feature_types, iterations_cap=iterations_cap,
    )

    y_proba_val_refit = catboost_model.predict_proba(model, X_val, feature_types)
    train_refit_val_auc = float(roc_auc_score(y_val.to_numpy(), y_proba_val_refit))

    model.save_model(str(MODELS_DIR / 'catboost_final.cbm'))

    best_threshold, _ = find_optimal_threshold(y_val.to_numpy(), y_proba_val_honest)

    update_json(METRICS_PATH, {
        MODEL_NAME: {
            'val': split_metrics(m_val),
            'optimal_threshold': best_threshold,
            'train_refit_val_auc': train_refit_val_auc,
            'train_refit_val_auc_note': (
                'val metric after final refit on train+val — '
                'this is a training-set score, not validation'
            ),
            'best_params': best_params,
            'n_trials': int(len(study.trials)),
            'fit_time_seconds': float(fit_seconds),
        }
    })

    print(
        f"\n[catboost] val ROC-AUC={m_val['roc_auc']:.4f} "
        f"PR-AUC={m_val['pr_auc']:.4f} F1={m_val['f1']:.4f} Brier={m_val['brier']:.4f} "
        f"(post-refit val AUC diagnostic={train_refit_val_auc:.4f})"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
