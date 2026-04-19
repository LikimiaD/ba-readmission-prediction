import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from catboost import CatBoostClassifier

from src.config import FIGURE_DPI, RANDOM_STATE
from src.features.build import DERIVED_FEATURE_NAMES, FeatureTypes
from src.models.catboost_model import _prepare_for_catboost, predict_proba

logger = logging.getLogger(__name__)


@dataclass
class ShapArtefacts:

    sample_size: int
    top_features: list[dict]


def _save(fig, path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor='white')
    plt.close(fig)


def run_shap(
    model: CatBoostClassifier, X_test, feature_types: FeatureTypes, out_dir,
    sample_size: int = 2000, top_k: int = 20, top_md_path=None,
) -> ShapArtefacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(sample_size, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)
    X_prepared = _prepare_for_catboost(X_sample, feature_types)

    logger.info('Computing SHAP values on %d rows', n)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prepared)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value = float(np.asarray(expected_value).ravel()[0])

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], expected_value),
        data=X_prepared.values,
        feature_names=list(X_prepared.columns),
    )

    shap.summary_plot(shap_values, X_prepared, show=False, plot_size=(10, 7))
    _save(plt.gcf(), out_dir / 'summary_beeswarm.png')

    shap.summary_plot(
        shap_values, X_prepared, plot_type='bar', show=False,
        max_display=top_k, plot_size=(10, 7),
    )
    _save(plt.gcf(), out_dir / 'summary_bar.png')

    probas = predict_proba(model, X_sample, feature_types)
    low_idx = int(np.argmin(probas))
    high_idx = int(np.argmax(probas))

    for local_idx, tag in ((low_idx, 'low_risk'), (high_idx, 'high_risk')):
        shap.plots.waterfall(explanation[local_idx], show=False, max_display=15)
        _save(plt.gcf(), out_dir / f'waterfall_{tag}.png')

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features = [
        {'name': str(X_prepared.columns[i]), 'mean_abs_shap': float(mean_abs[i])}
        for i in order[:top_k]
    ]

    top3 = order[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat_idx in zip(axes, top3):
        feat_name = X_prepared.columns[feat_idx]
        shap.dependence_plot(
            int(feat_idx), shap_values, X_prepared,
            ax=ax, show=False, interaction_index=None,
        )
        ax.set_title(f'Dependence — {feat_name}')
    _save(fig, out_dir / 'dependence_top3.png')

    if top_md_path is not None:
        _write_top_features_md(top_features, top_md_path, n)

    return ShapArtefacts(sample_size=n, top_features=top_features)


def _write_top_features_md(top_features, path, sample_size):
    derived = set(DERIVED_FEATURE_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'# SHAP top features (CatBoost, n={sample_size})',
        '',
        'Marker ✦ means the feature is derived in `src/features/build.py`; '
        'unmarked features come straight from the raw CSV.',
        '',
        '| rank | feature | origin | mean(\\|SHAP\\|) |',
        '|---:|---|:---:|---:|',
    ]
    for i, item in enumerate(top_features, start=1):
        marker = '✦' if item['name'] in derived else ''
        lines.append(
            f"| {i} | {item['name']} | {marker} | {item['mean_abs_shap']:.6f} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
