from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_dicts(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config(experiment_config, dataset_config=None, method_config=None, overrides=None):
    data = {}
    for path in [experiment_config, dataset_config, method_config]:
        if path:
            data = merge_dicts(data, load_yaml(path))
    if overrides:
        data = merge_dicts(data, overrides)
    return _to_namespace(data)


def load_mmcv_config(path):
    from mmcv import Config
    return Config.fromfile(str(Path(path)))
