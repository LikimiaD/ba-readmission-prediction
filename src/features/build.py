from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BINARY_TARGET_COL

DERIVED_FEATURE_NAMES = (
    'total_prior_visits',
    'emergency_ratio',
    'meds_per_day',
    'procedures_per_day',
    'labs_per_day',
    'had_emergency_visit',
    'frequent_inpatient',
    'n_distinct_diagnoses',
    'poorly_controlled_diabetes',
)

_DIAG_MISSING = 'Missing'


def build_derived_features(df):
    df = df.copy()

    inpatient = df['n_inpatient'].astype(float)
    outpatient = df['n_outpatient'].astype(float)
    emergency = df['n_emergency'].astype(float)
    time_hosp = df['time_in_hospital'].astype(float)

    df['total_prior_visits'] = inpatient + outpatient + emergency
    df['emergency_ratio'] = emergency / (inpatient + outpatient + emergency + 1.0)

    df['meds_per_day'] = df['n_medications'].astype(float) / (time_hosp + 1.0)
    df['procedures_per_day'] = df['n_procedures'].astype(float) / (time_hosp + 1.0)
    df['labs_per_day'] = df['n_lab_procedures'].astype(float) / (time_hosp + 1.0)

    df['had_emergency_visit'] = (emergency > 0).astype(int)
    df['frequent_inpatient'] = (inpatient >= 2).astype(int)

    diag_cols = [c for c in ('diag_1', 'diag_2', 'diag_3') if c in df.columns]
    if diag_cols:
        diag_matrix = df[diag_cols].astype(str).ne(_DIAG_MISSING)
        df['n_distinct_diagnoses'] = diag_matrix.sum(axis=1).astype(int)
    else:
        df['n_distinct_diagnoses'] = 0

    a1c = df['A1Ctest'].astype(str).str.lower() if 'A1Ctest' in df.columns else None
    dmed = df['diabetes_med'].astype(str).str.lower() if 'diabetes_med' in df.columns else None
    if a1c is not None and dmed is not None:
        df['poorly_controlled_diabetes'] = (a1c.eq('high') & dmed.eq('yes')).astype(int)
    else:
        df['poorly_controlled_diabetes'] = 0

    for c in DERIVED_FEATURE_NAMES:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


@dataclass(frozen=True)
class FeatureTypes:

    numeric: list[str]
    categorical: list[str]

    @property
    def all_features(self):
        return list(self.numeric) + list(self.categorical)


def split_features_by_type(df, target_col=BINARY_TARGET_COL):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric = []
    categorical = []
    for col in feature_cols:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric.append(col)
        else:
            categorical.append(col)
    return FeatureTypes(numeric=numeric, categorical=categorical)


def xy_split(df, target_col=BINARY_TARGET_COL):
    y = df[target_col].astype(int).copy()
    X = df.drop(columns=[target_col]).copy()
    return X, y
