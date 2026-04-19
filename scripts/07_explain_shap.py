import logging

from catboost import CatBoostClassifier

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.explain.shap_analysis import run_shap
from src.features.build import split_features_by_type, xy_split
from src.utils.io import load_parquet, update_json

SAMPLE_SIZE = 2000
log = logging.getLogger('shap')


def main():
    test = load_parquet(PROCESSED_DIR / 'test.parquet')
    X_test, _ = xy_split(test)
    feature_types = split_features_by_type(test)

    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / 'catboost_final.cbm'))

    artefacts = run_shap(
        model, X_test, feature_types,
        out_dir=FIGURES_DIR / 'shap',
        sample_size=SAMPLE_SIZE,
        top_k=20,
        top_md_path=REPORTS_DIR / 'shap_top_features.md',
    )

    update_json(METRICS_PATH, {
        'shap': {
            'sample_size': artefacts.sample_size,
            'top_features': artefacts.top_features,
        }
    })
    print("\n[shap] top features: " + ', '.join(f['name'] for f in artefacts.top_features[:5]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
