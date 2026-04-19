import logging
import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_STATE
from src.features.build import FeatureTypes

logger = logging.getLogger(__name__)


def _prepare_for_xgb(X, feature_types, categories=None):
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
    X_tr, cats = _prepare_for_xgb(X_train, feature_types)
    X_va, _ = _prepare_for_xgb(X_val, feature_types, categories=cats)
    dtrain = xgb.DMatrix(X_tr, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_va, label=y_val, enable_categorical=True)
    spw = _positive_class_weight(y_train)

    def objective(trial):
        params: dict[str, Any] = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'verbosity': 0,
            'scale_pos_weight': spw,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 50.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        booster = xgb.train(
            params, dtrain, num_boost_round=1200,
            evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False,
        )
        y_proba = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
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
    logger.info('XGBoost Optuna: %d trials in %.1fs, best AUC=%.4f',
                len(study.trials), dt, study.best_value)
    return study


def honest_val_predictions(
    best_params, X_train, y_train, X_val, y_val, feature_types: FeatureTypes,
) -> tuple[np.ndarray, int]:
    X_tr, cats = _prepare_for_xgb(X_train, feature_types)
    X_va, _ = _prepare_for_xgb(X_val, feature_types, categories=cats)
    dtrain = xgb.DMatrix(X_tr, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_va, label=y_val, enable_categorical=True)
    params = dict(best_params)
    params.update(
        objective='binary:logistic', eval_metric='auc', tree_method='hist',
        random_state=RANDOM_STATE, verbosity=0,
        scale_pos_weight=_positive_class_weight(y_train),
    )
    booster = xgb.train(
        params, dtrain, num_boost_round=1200,
        evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False,
    )
    best_iter = int(booster.best_iteration)
    y_proba = booster.predict(dval, iteration_range=(0, best_iter + 1))
    return y_proba, best_iter


def fit_final(
    best_params, X, y, feature_types: FeatureTypes, num_boost_round: int = 1200,
) -> tuple[xgb.Booster, dict, float]:
    X_prep, cats = _prepare_for_xgb(X, feature_types)
    dfull = xgb.DMatrix(X_prep, label=y, enable_categorical=True)
    params = dict(best_params)
    params.update(
        objective='binary:logistic', eval_metric='auc', tree_method='hist',
        random_state=RANDOM_STATE, verbosity=0,
        scale_pos_weight=_positive_class_weight(y),
    )
    t0 = time.perf_counter()
    booster = xgb.train(params, dfull, num_boost_round=num_boost_round)
    dt = time.perf_counter() - t0
    logger.info('XGBoost final fit in %.2fs (rounds=%d)', dt, num_boost_round)
    return booster, cats, dt


def predict_proba(booster, X, feature_types: FeatureTypes, categories) -> np.ndarray:
    X_prep, _ = _prepare_for_xgb(X, feature_types, categories=categories)
    dmat = xgb.DMatrix(X_prep, enable_categorical=True)
    return booster.predict(dmat)
