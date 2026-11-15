import random
from pathlib import Path

import numpy as np
import torch

from .io import ensure_dir, timestamp


ALL_CORRUPTIONS = [
    'segment_dropout',
    'segment_shuffle',
    'temporal_delay',
    'temporal_reverse',
    'camera_viewpoint',
    'nonrigid_joint_drift',
    'coordinate_quantization',
    'skeleton_mirror',
    'persistent_joint_missing',
    'joint_dropout_burst',
    'identity_switch',
    'left_right_asymmetry',
    'joint_topology_swap',
    'temporal_correlated_noise',
    'outlier_frames',
    'depth_axis_noise',
    'high_frequency_jitter',
    'mixed_failure',
]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_dataset_name(ann_file):
    ann = ann_file.lower()
    if 'ntu60' in ann:
        return 'NTU60'
    if 'ntu120' in ann:
        return 'NTU120'
    if 'k400' in ann or 'kinetics' in ann:
        return 'K400'
    if 'gym' in ann:
        return 'FineGYM'
    return Path(ann_file).stem


def infer_modality(cfg):
    for step in cfg.data.test.pipeline:
        if step.get('type') == 'GenSkeFeat':
            feats = step.get('feats', [])
            if feats:
                return '+'.join(feats)
    return 'unknown'


def infer_num_clips(cfg):
    for step in cfg.data.test.pipeline:
        if step.get('type') in ('UniformSample', 'UniformSampleDecode'):
            return step.get('num_clips', 1)
    return 1


def make_run_name(prefix, dataset, mode, seed):
    return f'{prefix}_{dataset}_{mode}_seed{seed}_{timestamp()}'


def prepare_paths(root, run_name):
    root = Path(root)
    results_root = ensure_dir(root / 'results')
    details_root = ensure_dir(results_root / 'details')
    paths = {
        'logs': ensure_dir(root / 'logs'),
        'results': results_root,
        'active': ensure_dir(details_root / 'active_runs' / run_name),
        'progress': ensure_dir(details_root / 'progress'),
        'pkl': ensure_dir(details_root / 'pkl' / run_name),
        'orders': ensure_dir(details_root / 'orders'),
        'legacy': ensure_dir(details_root / 'legacy'),
        'checkpoints_meta': ensure_dir(details_root / 'checkpoints_meta'),
    }
    return paths
