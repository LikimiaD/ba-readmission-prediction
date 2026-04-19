from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'

MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

for _d in (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    FIGURES_DIR / 'eda',
    FIGURES_DIR / 'roc',
    FIGURES_DIR / 'pr',
    FIGURES_DIR / 'calibration',
    FIGURES_DIR / 'confusion',
    FIGURES_DIR / 'shap',
):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

TARGET_COL = 'readmitted'
BINARY_TARGET_COL = 'readmitted_30d'
KAGGLE_DATASET = 'dubradave/hospital-readmissions'

METRICS_PATH = REPORTS_DIR / 'metrics.json'
TRAINING_LOG_PATH = REPORTS_DIR / 'training_log.txt'

FIGURE_DPI = 150
