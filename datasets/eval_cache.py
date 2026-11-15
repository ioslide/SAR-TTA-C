"""Lossless FP32 cache of the fixed offline evaluation pipeline.

EVAL_CACHE_DIR selects the local cache directory; an empty value disables it.
Files are memory-mapped read-only, so the OS loads/reclaims pages on demand.
Building a cache preserves RNG state so online adaptation follows the same draws.
"""
import fcntl
import hashlib
import json
import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import mmcv

from protogcn.datasets import BaseDataset, build_dataloader, build_dataset


class CachedEvalDataset(BaseDataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.keypoints = None
        self.class_prob = None
        super().__init__(str(self.directory / 'index.json'), [], test_mode=True,
                         start_index=0, modality='Pose')

    def load_annotations(self):
        metadata = json.loads(Path(self.ann_file).read_text())
        self.label_shape = metadata['label_shape']
        self.label_dtype = metadata['label_dtype']
        return [{'label': label} for label in metadata['labels']]

    def __getitem__(self, index):
        if self.keypoints is None:
            self.keypoints = np.load(self.directory / 'keypoints.npy', mmap_mode='r')
        return {
            'keypoint': torch.from_numpy(self.keypoints[index].copy()),
            'label': torch.from_numpy(np.asarray(
                self.video_infos[index]['label'], dtype=self.label_dtype).reshape(self.label_shape)),
        }


def _cache_key(dataset_cfg, loader_settings):
    annotation = Path(dataset_cfg['ann_file']).resolve()
    sources = [annotation]
    if annotation.name == 'index.json':
        sources.append(annotation.parent / json.loads(annotation.read_text())['data_file'])
    fingerprint = {
        'dataset': dataset_cfg,
        'loader': loader_settings,
        'versions': (torch.__version__, np.__version__, mmcv.__version__),
        'sources': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources],
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode())
    legacy = Path(__file__).resolve().parent.parent / 'old/protogcn/datasets'
    for path in [Path(__file__), legacy / 'base.py', legacy / 'pose_dataset.py',
                 legacy / 'builder.py'] + sorted((legacy / 'pipelines').glob('*.py')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_cache(directory, dataset_cfg, loader_settings):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        dataset = build_dataset(dataset_cfg, dict(test_mode=True))
        loader = build_dataloader(dataset, **loader_settings)
        labels = []
        keypoints = None
        for batch in loader:
            assert set(batch) == {'keypoint', 'label'}
            inputs = batch['keypoint'].numpy()
            assert inputs.dtype == np.float32
            if keypoints is None:
                label_shape = list(batch['label'].shape[1:])
                label_dtype = str(batch['label'].numpy().dtype)
                keypoints = np.lib.format.open_memmap(
                    directory / 'keypoints.npy', mode='w+', dtype=inputs.dtype,
                    shape=(len(dataset),) + inputs.shape[1:])
            start = len(labels)
            keypoints[start:start + len(inputs)] = inputs
            labels.extend(batch['label'].reshape(-1).tolist())
        assert len(labels) == len(dataset)
        assert labels == [int(item['label']) for item in dataset.video_infos]
        keypoints.flush()
        del keypoints
        (directory / 'index.json').write_text(json.dumps({
            'labels': labels, 'label_shape': label_shape, 'label_dtype': label_dtype}))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def cached_eval_dataset(dataset_cfg, loader_settings, cache_root):
    pipeline = dataset_cfg['pipeline']
    expected = ['PreNormalize3D', 'GenSkeFeat', 'UniformSampleDecode',
                'FormatGCNInput', 'Collect', 'ToTensor']
    if ([step['type'] for step in pipeline] != expected
            or dataset_cfg['type'] != 'PoseDataset'
            or pipeline[4]['keys'] != ['keypoint', 'label']
            or pipeline[4].get('meta_keys') != []):
        raise ValueError('Evaluation cache requires the fixed skeleton test pipeline')
    if (loader_settings['workers_per_gpu'] < 1 or loader_settings['shuffle']
            or loader_settings['drop_last']):
        raise ValueError('Evaluation cache requires workers >= 1, shuffle=False, drop_last=False')
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_cfg, loader_settings)
    directory = root / key
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not directory.exists():
            logging.getLogger('protogcn').info('Building FP32 evaluation cache: %s', directory)
            with TemporaryDirectory(prefix='.' + key + '-', dir=root) as temporary:
                _write_cache(Path(temporary), dataset_cfg, loader_settings)
                Path(temporary).rename(directory)
    logging.getLogger('protogcn').info('Using mmap evaluation cache: %s', directory)
    return CachedEvalDataset(directory)
