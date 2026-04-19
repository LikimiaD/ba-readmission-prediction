import logging

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import split_features_by_type, xy_split
from src.models import xgboost_model
from src.models.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion,
    plot_pr,
    plot_roc,
    split_metrics,
)
from src.utils.io import load_parquet, update_json

MODEL_NAME = 'xgboost'
N_TRIALS = 60
TIMEOUT_SECONDS = 900
log = logging.getLogger(MODEL_NAME)


def main():
    train = load_parquet(PROCESSED_DIR / 'train.parquet')
    val = load_parquet(PROCESSED_DIR / 'val.parquet')
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    feature_types = split_features_by_type(train)

    study = xgboost_model.tune(
        X_train, y_train, X_val, y_val, feature_types,
        n_trials=N_TRIALS, timeout_seconds=TIMEOUT_SECONDS,
    )
    best_params = dict(study.best_params)

    y_proba_val_honest, best_iter = xgboost_model.honest_val_predictions(
        best_params, X_train, y_train, X_val, y_val, feature_types,
    )
    m_val = compute_metrics(MODEL_NAME, 'val', y_val.to_numpy(), y_proba_val_honest)
    log.info('Honest val: %s (early-stopped iter=%d)', m_val, best_iter)

    plot_roc(y_val.to_numpy(), y_proba_val_honest, MODEL_NAME, FIGURES_DIR / 'roc' / 'xgboost.png')
    plot_pr(y_val.to_numpy(), y_proba_val_honest, MODEL_NAME, FIGURES_DIR / 'pr' / 'xgboost.png')
    plot_confusion(
        y_val.to_numpy(), y_proba_val_honest, MODEL_NAME,
        FIGURES_DIR / 'confusion' / 'xgboost.png',
    )

    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    scale = 1.0 + (len(X_val) / max(len(X_train), 1))
    num_boost_round = max(int(best_iter * scale), 1) if best_iter > 0 else 500
    booster, categories, fit_seconds = xgboost_model.fit_final(
        best_params, X_full, y_full, feature_types, num_boost_round=num_boost_round,
    )

    y_proba_refit = xgboost_model.predict_proba(booster, X_val, feature_types, categories)
    refit_val_auc = float(roc_auc_score(y_val.to_numpy(), y_proba_refit))

    best_threshold, _ = find_optimal_threshold(y_val.to_numpy(), y_proba_val_honest)

    joblib.dump(
        {'booster': booster, 'categories': categories},
        MODELS_DIR / 'xgboost_final.pkl', compress=3,
    )

    update_json(METRICS_PATH, {
        MODEL_NAME: {
            'val': split_metrics(m_val),
            'optimal_threshold': best_threshold,
            'train_refit_val_auc': refit_val_auc,
            'train_refit_val_auc_note': (
                'val AUC after refit on train+val — training-set score, not validation'
            ),
            'best_params': best_params,
            'n_trials': int(len(study.trials)),
            'fit_time_seconds': float(fit_seconds),
        }
    })
    print(
        f"\n[xgboost] val ROC-AUC={m_val['roc_auc']:.4f} "
        f"F1={m_val['f1']:.4f} Brier={m_val['brier']:.4f} "
        f"(refit-val AUC={refit_val_auc:.4f}, opt_thr={best_threshold:.2f})"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
