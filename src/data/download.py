import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import KAGGLE_DATASET, RAW_DIR

logger = logging.getLogger(__name__)


def download_dataset(dataset=KAGGLE_DATASET, dest=RAW_DIR, force=False):
    dest.mkdir(parents=True, exist_ok=True)
    has_csv = any(p.suffix.lower() == '.csv' for p in dest.iterdir())
    if has_csv and not force:
        return dest

    load_dotenv()
    if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
        raise RuntimeError(
            'Kaggle credentials missing. Put KAGGLE_USERNAME and KAGGLE_KEY '
            'into .env (see .env.example), or place hospital_readmissions.csv '
            'into data/raw/ manually.'
        )

    # Импорт отложен: без kaggle-креденшлов модуль всё равно бесполезен,
    # но сам src.data.download должен импортироваться.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest), unzip=True)

    n_files = sum(1 for p in dest.iterdir() if p.is_file())
    logger.info('Downloaded %s to %s (%d files)', dataset, dest, n_files)
    return dest
