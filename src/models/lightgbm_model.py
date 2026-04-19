import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_STATE
from src.features.build import FeatureTypes

logger = logging.getLogger(__name__)


def _prepare_for_lgbm(X, feature_types, categories=None):
    X = X.copy()
    dtype_map = {}
    for col in feature_types.categorical:
        if col not in X.columns:
            continue
        if categories and col in categories:
            X[col] = X[col].astype(str).astype(categories[col])
        else:
            cat = pd.CategoricalDtype(categories=sorted(X[col].astype(str).unique()))
            X[col] = X[col].astype(str).astype(cat)
            dtype_map[col] = cat
    return X, dtype_map


def _positive_class_weight(y):
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


def _objective_factory(X_train, y_train, X_val, y_val, feature_types):
    X_tr, cats = _prepare_for_lgbm(X_train, feature_types)
    X_va, _ = _prepare_for_lgbm(X_val, feature_types, categories=cats)

    cat_cols = [c for c in feature_types.categorical if c in X_tr.columns]
    train_ds = lgb.Dataset(
        X_tr, label=y_train, categorical_feature=cat_cols, free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        X_va, label=y_val, categorical_feature=cat_cols,
        reference=train_ds, free_raw_data=False,
    )
    spw = _positive_class_weight(y_train)

    def objective(trial):
        params: dict[str, Any] = {
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'random_state': RANDOM_STATE,
            'scale_pos_weight': spw,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        }
        booster = lgb.train(
            params, train_ds, num_boost_round=1200, valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
        )
        y_proba = booster.predict(X_va)
        return float(roc_auc_score(y_val, y_proba))

    return objective


def tune(
    X_train, y_train, X_val, y_val, feature_types: FeatureTypes,
    n_trials: int = 60, timeout_seconds: int = 900,
) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    objective = _objective_factory(X_train, y_train, X_val, y_val, feature_types)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    dt = time.perf_counter() - t0
    logger.info('LightGBM Optuna: %d trials in %.1fs, best AUC=%.4f',
                len(study.trials), dt, study.best_value)
    return study


def honest_val_predictions(
    best_params, X_train, y_train, X_val, y_val, feature_types: FeatureTypes,
) -> tuple[np.ndarray, int]:
    X_tr, cats = _prepare_for_lgbm(X_train, feature_types)
    X_va, _ = _prepare_for_lgbm(X_val, feature_types, categories=cats)
    cat_cols = [c for c in feature_types.categorical if c in X_tr.columns]
    train_ds = lgb.Dataset(
        X_tr, label=y_train, categorical_feature=cat_cols, free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        X_va, label=y_val, categorical_feature=cat_cols,
        reference=train_ds, free_raw_data=False,
    )

    params = dict(best_params)
    params.update(
        objective='binary', metric='auc', verbose=-1,
        random_state=RANDOM_STATE, scale_pos_weight=_positive_class_weight(y_train),
    )
    booster = lgb.train(
        params, train_ds, num_boost_round=1200, valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    return booster.predict(X_va), int(booster.best_iteration or booster.current_iteration())


def fit_final(
    best_params, X, y, feature_types: FeatureTypes, num_boost_round: int = 1200,
) -> tuple[lgb.Booster, dict, float]:
    X_prep, cats = _prepare_for_lgbm(X, feature_types)
    cat_cols = [c for c in feature_types.categorical if c in X_prep.columns]
    full_ds = lgb.Dataset(X_prep, label=y, categorical_feature=cat_cols)
    params = dict(best_params)
    params.update(
        objective='binary', metric='auc', verbose=-1,
        random_state=RANDOM_STATE, scale_pos_weight=_positive_class_weight(y),
    )
    t0 = time.perf_counter()
    booster = lgb.train(params, full_ds, num_boost_round=num_boost_round)
    dt = time.perf_counter() - t0
    logger.info('LightGBM final fit in %.2fs (rounds=%d)', dt, num_boost_round)
    return booster, cats, dt


def predict_proba(booster, X, feature_types: FeatureTypes, categories) -> np.ndarray:
    X_prep, _ = _prepare_for_lgbm(X, feature_types, categories=categories)
    return booster.predict(X_prep)
