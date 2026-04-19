import json
from pathlib import Path

import pandas as pd


def save_json(payload, path, *, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False, default=_json_default)
        fh.write('\n')


def load_json(path):
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def update_json(path, patch):
    data = {}
    if path.exists():
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            data = {}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            merged = dict(data[k])
            merged.update(v)
            data[k] = merged
        else:
            data[k] = v
    save_json(data, path)
    return data


def save_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path):
    return pd.read_parquet(path)


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
