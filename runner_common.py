import os
import shutil
from pathlib import Path

import torch

from core.config import load_mmcv_config, load_runtime_config
from core.io import dump_json, setup_logger
from core.runtime import make_run_name, prepare_paths, set_global_seed


def add_runtime_arguments(parser, method_required=False):
    parser.add_argument('--experiment-config', required=True)
    parser.add_argument('--dataset-config', required=True)
    parser.add_argument('--method-config')
    parser.add_argument('--base-config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--work-dir', default='.')
    parser.add_argument('--method', required=method_required)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--severity', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--note', default=None)
    parser.add_argument('--dry-run', action='store_true')


def build_runtime(args, mode, method_config=None):
    method_config = method_config or args.method_config
    cfg = load_runtime_config(args.experiment_config, args.dataset_config, method_config)
    cfg.experiment.mode = mode
    if args.method:
        cfg.method.name = args.method
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.severity is not None:
        cfg.experiment.severity = args.severity
    if args.batch_size is not None:
        cfg.experiment.batch_size = args.batch_size
    if args.checkpoint:
        cfg.experiment.checkpoint = args.checkpoint
    if args.base_config:
        cfg.experiment.source_config = args.base_config
    if args.note is not None:
        cfg.experiment.note = args.note
    elif not hasattr(cfg.experiment, 'note'):
        cfg.experiment.note = ''
    return cfg


def average_clip(test_cfg, cls_score):
    average_clips = test_cfg.get('average_clips', 'prob')
    if average_clips == 'prob':
        return torch.softmax(cls_score, dim=-1).mean(dim=-2, keepdim=True)
    if average_clips == 'score':
        return cls_score.mean(dim=-2, keepdim=True)
    return cls_score


def evaluate_on_loader(adapt_model, data_loader, test_cfg):
    outputs = []
    for data in data_loader:
        keypoint = data['keypoint'].cuda(non_blocking=True)
        batch_size, num_clips = keypoint.shape[:2]
        keypoint = keypoint.reshape((batch_size * num_clips,) + keypoint.shape[2:])
        logits = adapt_model(keypoint)
        logits = logits.detach().reshape(batch_size, num_clips, logits.shape[-1])
        outputs.append(average_clip(test_cfg, logits).squeeze(1))
    return list(torch.cat(outputs).cpu().numpy())


def extract_top1(metrics):
    if 'top_k_accuracy' in metrics:
        return metrics['top_k_accuracy'][0]
    if 'top1_acc' in metrics:
        return metrics['top1_acc']
    raise KeyError(f'Unsupported metric keys: {list(metrics.keys())}')


def resolve_offline_annotation_file(corruption, cfg):
    dataset = cfg.dataset
    directory = dataset.offline_corruption_dir
    source = dataset.ann_file
    severity = getattr(dataset, 'offline_corruption_severity', 3)
    seed = getattr(dataset, 'offline_corruption_seed', 42)
    ann_file = Path(directory) / f'{Path(source).stem}_corrupt_{corruption}_s{severity}_seed{seed}.pkl'
    mmap_index = Path(f'{ann_file}.mmap') / 'index.json'
    if mmap_index.is_file():
        return str(mmap_index)
    if ann_file.is_file():
        return str(ann_file)
    raise FileNotFoundError(f'offline corruption dataset not found: {ann_file}')


def load_model_adapter(method_name, args, cfg, mmcv_cfg, method_config=None):
    from methods import setup_method
    from models import build_model

    method_cfg = load_runtime_config(
        args.experiment_config,
        args.dataset_config,
        method_config or f'configs/method/{method_name}.yaml',
    )
    method_cfg.experiment.conda_env = cfg.experiment.conda_env
    method_cfg.experiment.note = cfg.experiment.note
    method_cfg.NOTE = getattr(cfg.experiment, 'note', '')
    model = build_model(mmcv_cfg.model, cfg.experiment.checkpoint)
    return setup_method(method_name, model, method_cfg)[0]


def build_domain_loader(mmcv_cfg, cfg, corruption_type, corruption_tag, seed, offline):
    from datasets import build_cfg_for_corruption, build_dataset_and_loader

    eval_cfg = build_cfg_for_corruption(
        mmcv_cfg,
        None if offline else corruption_type,
        cfg.experiment.severity,
        cfg.experiment.batch_size,
    )
    if offline:
        eval_cfg.data.test.ann_file = resolve_offline_annotation_file(corruption_tag, cfg)
    cache_dir = os.environ.get('EVAL_CACHE_DIR', '/tmp/protogcn-eval-cache') if offline else None
    dataset, data_loader = build_dataset_and_loader(eval_cfg, seed, cache_dir=cache_dir)
    return eval_cfg, dataset, data_loader


def load_mmcv_test_config(cfg):
    mmcv_cfg = load_mmcv_config(cfg.experiment.source_config)
    source_ann_file = getattr(cfg.dataset, 'ann_file', None)
    if source_ann_file:
        mmcv_cfg.data.test.ann_file = source_ann_file
    return mmcv_cfg


def initialize_run(args, cfg):
    dataset_name = getattr(cfg.dataset, 'name', 'dataset')
    run_name = make_run_name(
        Path(cfg.experiment.source_config).stem,
        dataset_name,
        cfg.experiment.mode,
        cfg.experiment.seed,
    )
    paths = prepare_paths(args.work_dir, run_name)
    logger = setup_logger(paths['logs'] / f'{run_name}.log')
    legacy_migrate(args.work_dir, paths, logger)
    return run_name, paths, logger


def result_csv_path(args, cfg):
    """Return the summary CSV scoped to this run's work directory."""
    dataset_name = result_dataset_name(args, cfg)
    mode = getattr(cfg.experiment, 'mode', 'benchmark')
    return Path(args.work_dir).resolve() / 'results' / f'{dataset_name}_{mode}.csv'


def result_dataset_name(args, cfg):
    """Return the stable dataset identifier used in result metadata and paths."""
    dataset_config = getattr(args, 'dataset_config', None)
    if dataset_config:
        return Path(dataset_config).stem
    return getattr(cfg.dataset, 'name', 'dataset')


def write_dry_run(run_name, paths, logger, payload):
    payload = {'status': 'dry-run-ready', **payload}
    progress_path = paths['progress'] / f'{run_name}.json'
    dump_json(progress_path, payload)
    logger.info('dry-run plan written to %s', progress_path)


def legacy_migrate(root, paths, logger):
    legacy_targets = ['result.pkl', 'benchmark_progress.json', 'benchmark_summary.csv']
    legacy_targets.extend(path.name for path in Path(root).glob('result_*.pkl'))
    for name in legacy_targets:
        source = Path(root) / name
        if not source.exists():
            continue
        destination = paths['legacy'] / name
        if destination.exists():
            continue
        logger.info('migrate legacy artifact %s -> %s', source, destination)
        shutil.move(str(source), str(destination))


__all__ = [
    'add_runtime_arguments',
    'average_clip',
    'build_domain_loader',
    'build_runtime',
    'evaluate_on_loader',
    'extract_top1',
    'initialize_run',
    'load_model_adapter',
    'load_mmcv_test_config',
    'resolve_offline_annotation_file',
    'result_dataset_name',
    'result_csv_path',
    'set_global_seed',
    'write_dry_run',
]
