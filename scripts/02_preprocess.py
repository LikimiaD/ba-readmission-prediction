import logging
import sys
from datetime import datetime, timezone

from src.config import BINARY_TARGET_COL, METRICS_PATH, RANDOM_STATE
from src.data.preprocess import run_preprocess
from src.features.build import DERIVED_FEATURE_NAMES
from src.utils.io import load_parquet, update_json

log = logging.getLogger('02_preprocess')


def main():
    paths = run_preprocess()
    train = load_parquet(paths['train'])
    val = load_parquet(paths['val'])
    test = load_parquet(paths['test'])

    n_features_after = train.shape[1] - 1
    n_features_before = n_features_after - len(DERIVED_FEATURE_NAMES)
    meta = {
        'dataset': 'dubradave/hospital-readmissions',
        'random_state': RANDOM_STATE,
        'train_size': int(len(train)),
        'val_size': int(len(val)),
        'test_size': int(len(test)),
        'positive_rate_train': float(train[BINARY_TARGET_COL].mean()),
        'positive_rate_val': float(val[BINARY_TARGET_COL].mean()),
        'positive_rate_test': float(test[BINARY_TARGET_COL].mean()),
        'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
        'run_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'feature_engineering': list(DERIVED_FEATURE_NAMES),
        'n_features_before': int(n_features_before),
        'n_features_after': int(n_features_after),
    }
    update_json(METRICS_PATH, {'meta': meta})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
