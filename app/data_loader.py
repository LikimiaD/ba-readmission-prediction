"""Одноразовая загрузка модели, теста, SHAP-кеша и metrics.json"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import FeatureTypes, split_features_by_type, xy_split
from src.models.catboost_model import _prepare_for_catboost, predict_proba

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / 'catboost_final.cbm'
TEST_PATH = PROCESSED_DIR / 'test.parquet'
SHAP_CACHE_PATH = PROCESSED_DIR / 'shap_test.npy'
SHAP_BASE_PATH = PROCESSED_DIR / 'shap_test_base.npy'

RISK_BAND_HALF_WIDTH = 0.10


@dataclass
class ArtifactBundle:
    model: CatBoostClassifier
    feature_types: FeatureTypes
    X_test: pd.DataFrame
    y_test: pd.Series
    proba_test: np.ndarray
    shap_values: np.ndarray
    shap_base_value: float
    metrics: dict
    threshold: float
    feature_names: list[str]

    @property
    def n_patients(self):
        return int(len(self.X_test))

    def risk_band(self, p):
        lo = max(0.0, self.threshold - RISK_BAND_HALF_WIDTH)
        hi = min(1.0, self.threshold + RISK_BAND_HALF_WIDTH)
        if p < lo:
            return 'low'
        if p >= hi:
            return 'high'
        return 'medium'


def _artefacts_present():
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])))
    if not TEST_PATH.exists():
        missing.append(str(TEST_PATH.relative_to(TEST_PATH.parents[2])))
    if not METRICS_PATH.exists():
        missing.append(str(METRICS_PATH.relative_to(METRICS_PATH.parents[1])))
    return missing


def artefacts_ready() -> bool:
    return not _artefacts_present()


def missing_artefacts_message() -> str:
    missing = _artefacts_present()
    if not missing:
        return ''
    return (
        'Artefacts are missing: ' + ', '.join(missing) + '. '
        'Run `make all` in the repository root to regenerate them.'
    )


def _load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))


def _resolve_threshold(metrics):
    cat = metrics.get('catboost') or {}
    thr = cat.get('optimal_threshold')
    if thr is not None:
        return float(thr)
    for name in ('lightgbm', 'xgboost', 'random_forest', 'baseline'):
        block = metrics.get(name) or {}
        if 'optimal_threshold' in block:
            return float(block['optimal_threshold'])
    return 0.5


def _compute_or_load_shap(model, X_test, feature_types):
    if SHAP_CACHE_PATH.exists() and SHAP_BASE_PATH.exists():
        shap_values = np.load(SHAP_CACHE_PATH)
        base_value = float(np.load(SHAP_BASE_PATH).item())
        if shap_values.shape[0] == len(X_test):
            return shap_values, base_value
        logger.warning(
            'Cached SHAP shape %s mismatches test length %d — recomputing',
            shap_values.shape, len(X_test),
        )

    import shap

    logger.info('Computing SHAP values for %d patients', len(X_test))
    X_prepared = _prepare_for_catboost(X_test, feature_types)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prepared)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value = float(np.asarray(expected_value).ravel()[0])
    else:
        expected_value = float(expected_value)

    SHAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(SHAP_CACHE_PATH, shap_values)
    np.save(SHAP_BASE_PATH, np.array([expected_value]))
    return shap_values, float(expected_value)


def load_artifacts() -> ArtifactBundle:
    missing = _artefacts_present()
    if missing:
        raise FileNotFoundError('missing artefacts: ' + ', '.join(missing))

    logger.info('Loading CatBoost model from %s', MODEL_PATH)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    test = pd.read_parquet(TEST_PATH)
    X_test, y_test = xy_split(test)
    feature_types = split_features_by_type(test)

    proba_test = predict_proba(model, X_test, feature_types)
    shap_values, base_value = _compute_or_load_shap(model, X_test, feature_types)

    metrics = _load_metrics()
    threshold = _resolve_threshold(metrics)

    feature_names = list(_prepare_for_catboost(X_test, feature_types).columns)
    return ArtifactBundle(
        model=model,
        feature_types=feature_types,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        proba_test=proba_test,
        shap_values=shap_values,
        shap_base_value=base_value,
        metrics=metrics,
        threshold=threshold,
        feature_names=feature_names,
    )
