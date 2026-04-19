import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_STATE
from src.features.build import FeatureTypes

logger = logging.getLogger(__name__)


def build_pipeline(feature_types: FeatureTypes) -> Pipeline:
    numeric_pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
    categorical_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipe, list(feature_types.numeric)),
        ('cat', categorical_pipe, list(feature_types.categorical)),
    ])
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def fit(pipeline, X, y):
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    dt = time.perf_counter() - t0
    logger.info('Random forest fit in %.2fs', dt)
    return pipeline, dt


def predict_proba(pipeline, X):
    return pipeline.predict_proba(X)[:, 1]
