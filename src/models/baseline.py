import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.features.build import FeatureTypes

logger = logging.getLogger(__name__)


def build_pipeline(feature_types: FeatureTypes) -> Pipeline:
    numeric_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
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
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])


def fit(pipeline, X, y):
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    dt = time.perf_counter() - t0
    logger.info('Logistic regression fit in %.2fs', dt)
    return pipeline, dt


def predict_proba(pipeline, X):
    return pipeline.predict_proba(X)[:, 1]
