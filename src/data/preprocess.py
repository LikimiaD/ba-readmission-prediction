import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BINARY_TARGET_COL,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from src.features.build import DERIVED_FEATURE_NAMES, build_derived_features
from src.utils.io import save_parquet

logger = logging.getLogger(__name__)

LEAKAGE_COLUMNS = ('encounter_id', 'patient_nbr')

QMARK_COLUMNS = (
    'medical_specialty',
    'race',
    'diag_1',
    'diag_2',
    'diag_3',
    'payer_code',
    'weight',
)


def find_raw_csv(raw_dir=RAW_DIR):
    csvs = sorted(raw_dir.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No CSV files in {raw_dir}. Did you run the download step?')
    return csvs[0]


def normalize_missing(df):
    for col in QMARK_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace('?', np.nan)
    return df


def drop_leakage_columns(df):
    to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop)


def derive_binary_target(df):
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found.")

    uniq = set(df[TARGET_COL].dropna().astype(str).str.lower().unique())
    if {'<30', '>30', 'no'}.issubset(uniq) or '<30' in uniq:
        positives = df[TARGET_COL].astype(str).eq('<30')
    elif uniq.issubset({'yes', 'no'}):
        positives = df[TARGET_COL].astype(str).str.lower().eq('yes')
        logger.warning(
            "Target column only has yes/no — using any-readmission as 30-day proxy."
        )
    else:
        raise ValueError(f'Unexpected readmitted values: {sorted(uniq)}')

    df = df.copy()
    df[BINARY_TARGET_COL] = positives.astype(int)
    df = df.drop(columns=[TARGET_COL])
    return df


def stratified_three_way_split(
    df,
    target_col=BINARY_TARGET_COL,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    random_state=RANDOM_STATE,
):
    y = df[target_col]
    train_and_val, test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state,
    )
    train, val = train_test_split(
        train_and_val,
        test_size=val_size,
        stratify=train_and_val[target_col],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def run_preprocess(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    csv_path = find_raw_csv(raw_dir)
    logger.info('Reading %s', csv_path.name)
    df = pd.read_csv(csv_path)

    df = normalize_missing(df)
    df = drop_leakage_columns(df)
    df = derive_binary_target(df)
    df = build_derived_features(df)

    train, val, test = stratified_three_way_split(df)

    paths = {
        'train': processed_dir / 'train.parquet',
        'val': processed_dir / 'val.parquet',
        'test': processed_dir / 'test.parquet',
    }
    for name, frame in (('train', train), ('val', val), ('test', test)):
        save_parquet(frame, paths[name])
        logger.info('Saved %s split: n=%d', name, len(frame))

    return paths
