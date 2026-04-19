import logging
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from src.features.build import FeatureTypes
from src.models.catboost_model import _prepare_for_catboost

logger = logging.getLogger(__name__)


class _CatBoostSklearnShim(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, feature_types):
        self.base_model = base_model
        self.feature_types = feature_types
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X_prepared = _prepare_for_catboost(X, self.feature_types)
        return self.base_model.predict_proba(X_prepared)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_isotonic(
    base_model: CatBoostClassifier, X_cal, y_cal, feature_types: FeatureTypes,
) -> tuple[CalibratedClassifierCV, float]:
    shim = _CatBoostSklearnShim(base_model, feature_types)
    calibrator = CalibratedClassifierCV(estimator=shim, method='isotonic', cv='prefit')
    t0 = time.perf_counter()
    calibrator.fit(X_cal, y_cal)
    dt = time.perf_counter() - t0
    logger.info('Isotonic calibration fit in %.2fs', dt)
    return calibrator, dt


def predict_calibrated(calibrator, X):
    return calibrator.predict_proba(X)[:, 1]
