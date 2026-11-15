from copy import deepcopy

from protogcn.datasets import build_dataloader as build_protogcn_dataloader
from protogcn.datasets import build_dataset as build_protogcn_dataset


def build_cfg_for_corruption(cfg, corruption_type=None, severity=None, batch_size=None):
    cfg = deepcopy(cfg)
    cfg.data.test.test_mode = True
    if corruption_type:
        cfg.data.test.pipeline.insert(1, dict(type=corruption_type, severity=severity))
    if batch_size is not None:
        cfg.data.videos_per_gpu = batch_size
        cfg.data.setdefault('test_dataloader', {})['videos_per_gpu'] = batch_size
    return cfg


def build_dataset_and_loader(cfg, seed, cache_dir=None):
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False,
        seed=seed,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
    )
    if dataloader_setting['workers_per_gpu'] > 0:
        dataloader_setting['prefetch_factor'] = 4
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    if cache_dir:
        from .eval_cache import cached_eval_dataset
        dataset = cached_eval_dataset(cfg.data.test, dataloader_setting, cache_dir)
    else:
        dataset = build_protogcn_dataset(cfg.data.test, dict(test_mode=True))
    return dataset, build_protogcn_dataloader(dataset, **dataloader_setting)
