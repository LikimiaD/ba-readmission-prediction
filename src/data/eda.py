import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RAW_DIR, REPORTS_DIR
from src.data.preprocess import (
    derive_binary_target,
    drop_leakage_columns,
    find_raw_csv,
    normalize_missing,
)
from src.features.build import DERIVED_FEATURE_NAMES, build_derived_features

logger = logging.getLogger(__name__)

EDA_FIG_DIR = FIGURES_DIR / 'eda'

_DERIVED_FEATURE_NOTES = {
    'total_prior_visits': 'sum of n_inpatient + n_outpatient + n_emergency — stable `medicalisation` proxy',
    'emergency_ratio': 'share of emergency visits among all prior contacts — marker of unstable disease course',
    'meds_per_day': 'n_medications / (time_in_hospital + 1) — treatment intensity per day of stay',
    'procedures_per_day': 'n_procedures / (time_in_hospital + 1) — procedural intensity per day',
    'labs_per_day': 'n_lab_procedures / (time_in_hospital + 1) — lab intensity per day',
    'had_emergency_visit': 'binary flag: any prior emergency visit — linear-model friendly',
    'frequent_inpatient': 'binary flag: ≥2 prior inpatient stays — known clinical predictor',
    'n_distinct_diagnoses': 'count of non-missing diag_1..diag_3 — multi-diagnosis complexity',
    'poorly_controlled_diabetes': 'A1C=high AND on diabetes meds — canonical risk factor',
}


def _save(fig, path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def run_eda(raw_dir=RAW_DIR, summary_path=None):
    sns.set_theme(style='whitegrid')
    EDA_FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = summary_path or (REPORTS_DIR / 'eda_summary.md')

    csv_path = find_raw_csv(raw_dir)
    df = pd.read_csv(csv_path)

    df_norm = normalize_missing(df.copy())
    df_clean = drop_leakage_columns(df_norm)
    df_bin = derive_binary_target(df_clean)
    df_bin = build_derived_features(df_bin)

    missing = df_norm.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if len(missing):
        fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.4)))
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color='steelblue')
        ax.set_xlabel('Share of NaN')
        ax.set_ylabel('Column')
        ax.set_title('Missing-value share by column')
        _save(fig, EDA_FIG_DIR / 'missing_values.png')

    pos_rate = float(df_bin['readmitted_30d'].mean())
    counts = df_bin['readmitted_30d'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color='steelblue')
    ax.set_title(f'Target class balance (positive rate = {pos_rate:.3f})')
    ax.set_xlabel('readmitted_30d')
    ax.set_ylabel('Count')
    _save(fig, EDA_FIG_DIR / 'target_balance.png')

    numeric_cols = [
        c for c in df_bin.select_dtypes(include=[np.number]).columns if c != 'readmitted_30d'
    ]
    if numeric_cols:
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 3 * rows))
        axes = np.atleast_2d(axes).ravel()
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df_bin[col].dropna(), bins=30, ax=ax, color='steelblue')
            ax.set_title(col)
        for ax in axes[len(numeric_cols):]:
            ax.axis('off')
        fig.suptitle('Numeric feature distributions', y=1.02)
        _save(fig, EDA_FIG_DIR / 'numeric_distributions.png')

    if len(numeric_cols) >= 2:
        corr = df_bin[numeric_cols + ['readmitted_30d']].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Numeric correlation matrix')
        _save(fig, EDA_FIG_DIR / 'correlation_matrix.png')

    if 'age' in df_bin.columns:
        age_rate = df_bin.groupby('age')['readmitted_30d'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=age_rate.index.astype(str), y=age_rate.values, ax=ax, color='steelblue')
        ax.set_title('Positive rate by age bucket')
        ax.set_xlabel('age')
        ax.set_ylabel('P(readmit_30d = 1)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        _save(fig, EDA_FIG_DIR / 'target_by_age.png')

    inpatient_col = next(
        (c for c in ['n_inpatient', 'n_previous_admissions'] if c in df_bin.columns), None
    )
    if inpatient_col:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_bin, x='readmitted_30d', y=inpatient_col, ax=ax)
        ax.set_title(f'{inpatient_col} vs target')
        _save(fig, EDA_FIG_DIR / 'target_by_inpatient.png')

    if 'diag_1' in df_bin.columns:
        top_diag = df_bin['diag_1'].value_counts().head(10).index
        diag_df = df_bin[df_bin['diag_1'].isin(top_diag)]
        diag_rate = diag_df.groupby('diag_1')['readmitted_30d'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=diag_rate.values, y=diag_rate.index, ax=ax, color='steelblue')
        ax.set_title('Positive rate by top-10 primary diagnoses')
        ax.set_xlabel('P(readmit_30d = 1)')
        _save(fig, EDA_FIG_DIR / 'target_by_diagnosis.png')

    missing_list = list(missing.index) if len(missing) else 'none'
    derived_lines = '\n'.join(
        f'- `{name}`: {_DERIVED_FEATURE_NOTES[name]}' for name in DERIVED_FEATURE_NAMES
    )
    summary = f"""# EDA summary

Source: `{csv_path.name}` in `data/raw/`.

- Shape: {df.shape[0]:,} rows, {df.shape[1]} columns.
- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB.
- Positive rate (`readmitted_30d=1`): **{pos_rate:.4f}**.
- Columns with missing values after `'?'` normalization: {missing_list}.

===
{derived_lines}
"""
    summary_path.write_text(summary, encoding='utf-8')
    logger.info('EDA summary written to %s', summary_path)
    return summary_path
