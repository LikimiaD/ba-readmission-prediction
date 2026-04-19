import logging
import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_STATE
from src.features.build import FeatureTypes

logger = logging.getLogger(__name__)


def _prepare_for_catboost(X, feature_types):
    X = X.copy()
    for col in feature_types.categorical:
        X[col] = X[col].astype(object).where(X[col].notna(), 'missing').astype(str)
    return X


def build_pool(X, y, feature_types: FeatureTypes) -> Pool:
    X_prepared = _prepare_for_catboost(X, feature_types)
    return Pool(data=X_prepared, label=y, cat_features=list(feature_types.categorical))


def _objective_factory(X_train, y_train, X_val, y_val, feature_types):
    train_pool = build_pool(X_train, y_train, feature_types)
    val_pool = build_pool(X_val, y_val, feature_types)

    def objective(trial):
        params: dict[str, Any] = {
            'iterations': trial.suggest_int('iterations', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 12.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 80),
            'grow_policy': trial.suggest_categorical(
                'grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'],
            ),
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 12),
        }
        if params['grow_policy'] == 'SymmetricTree':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        else:
            params['bootstrap_type'] = 'Bernoulli'
            params['subsample'] = trial.suggest_float('subsample', 0.7, 1.0)

        model = CatBoostClassifier(
            **params,
            loss_function='Logloss',
            eval_metric='AUC',
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=50)
        y_proba = model.predict_proba(val_pool)[:, 1]
        return float(roc_auc_score(y_val, y_proba))

    return objective


def tune(
    X_train, y_train, X_val, y_val, feature_types: FeatureTypes,
    n_trials: int = 150, timeout_seconds: int = 1800,
) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    objective = _objective_factory(X_train, y_train, X_val, y_val, feature_types)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    dt = time.perf_counter() - t0
    logger.info(
        'Optuna: %d trials in %.1fs, best AUC=%.4f',
        len(study.trials), dt, study.best_value,
    )
    return study


def _full_params_from(best_params):
    params = dict(best_params)
    if params.get('grow_policy') and params['grow_policy'] != 'SymmetricTree':
        params.setdefault('bootstrap_type', 'Bernoulli')
    return params


def honest_val_predictions(
    best_params, X_train, y_train, X_val, y_val, feature_types: FeatureTypes,
) -> tuple[np.ndarray, int]:
    train_pool = build_pool(X_train, y_train, feature_types)
    val_pool = build_pool(X_val, y_val, feature_types)
    model = CatBoostClassifier(
        **_full_params_from(best_params),
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=50)
    best_iteration = int(getattr(model, 'tree_count_', None) or model.get_best_iteration() or 0)
    return model.predict_proba(val_pool)[:, 1], best_iteration


def fit_final(
    best_params, X, y, feature_types: FeatureTypes, iterations_cap: int | None = None,
) -> tuple[CatBoostClassifier, float]:
    params = _full_params_from(best_params)
    if iterations_cap and iterations_cap > 0:
        params['iterations'] = int(iterations_cap)
    pool = build_pool(X, y, feature_types)
    model = CatBoostClassifier(
        **params,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    model.fit(pool)
    dt = time.perf_counter() - t0
    logger.info('CatBoost final fit in %.2fs (iterations=%d)', dt, int(params['iterations']))
    return model, dt


def predict_proba(model: CatBoostClassifier, X, feature_types: FeatureTypes) -> np.ndarray:
    X_prepared = _prepare_for_catboost(X, feature_types)
    return model.predict_proba(X_prepared)[:, 1]
