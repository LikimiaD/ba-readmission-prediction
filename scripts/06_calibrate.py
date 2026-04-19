import logging

import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, PROCESSED_DIR
from src.features.build import split_features_by_type, xy_split
from src.models import catboost_model
from src.models.calibration import calibrate_isotonic, predict_calibrated
from src.models.evaluate import plot_calibration
from src.utils.io import load_parquet, update_json

log = logging.getLogger('calibrate')


def main():
    val = load_parquet(PROCESSED_DIR / 'val.parquet')
    X_val, y_val = xy_split(val)
    feature_types = split_features_by_type(val)

    base = CatBoostClassifier()
    base.load_model(str(MODELS_DIR / 'catboost_final.cbm'))

    y_proba_before = catboost_model.predict_proba(base, X_val, feature_types)
    brier_before = float(brier_score_loss(y_val.to_numpy(), y_proba_before))

    calibrator, _ = calibrate_isotonic(base, X_val, y_val, feature_types)
    y_proba_after = predict_calibrated(calibrator, X_val)
    brier_after = float(brier_score_loss(y_val.to_numpy(), y_proba_after))

    plot_calibration(
        y_val.to_numpy(), y_proba_before,
        'catboost (uncalibrated)', FIGURES_DIR / 'calibration' / 'before.png',
    )
    plot_calibration(
        y_val.to_numpy(), y_proba_after,
        'catboost (isotonic)', FIGURES_DIR / 'calibration' / 'after.png',
    )

    joblib.dump(calibrator, MODELS_DIR / 'catboost_calibrated.pkl')
    log.info('Isotonic calibrator saved (val Brier %.4f -> %.4f)', brier_before, brier_after)

    update_json(METRICS_PATH, {
        'calibration': {
            'method': 'isotonic',
            'brier_val_before': brier_before,
            'brier_val_after': brier_after,
        }
    })
    print(
        f"\n[calibration] val brier_before={brier_before:.4f} "
        f"brier_after={brier_after:.4f} (status finalised in evaluate step)"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
