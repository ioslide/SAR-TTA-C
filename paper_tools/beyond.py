"""Single-pass experiments newly specified in the paper; isolated result roots.

Labels are used only by the stream sampler and evaluator, never by adapters.
Dirichlet ordering follows BIT-DA/RoTTA/core/data/ttasampler.py. No existing
corruption, normalization, batch size, learning rate, or main-stream order changes.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import build_cfg_for_corruption, build_dataset_and_loader
from methods.PaperExtensions import roll, setup_theory
from models import build_model
from runner_common import (add_runtime_arguments, build_runtime, build_domain_loader,
                           evaluate_on_loader, load_mmcv_test_config, load_model_adapter,
                           set_global_seed)
from paper_tools.collect_runs import CORRUPTIONS
from paper_tools.paper_logging import GpuTimer, StreamLogger

LABEL_PROTOCOLS = ('dirichlet1', 'dirichlet0.1', 'dirichlet0.01', 'lt10')
ROLL_PROTOCOLS = ('roll90', 'roll180', 'roll180_reverse')
THEORY_PROTOCOLS = ('theory', 'theory_decomposition', 'exact_gate')


def label_order(labels, protocol, seed):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.RandomState(seed)
    if protocol == 'lt10':
        maximum = min(int((labels == c).sum()) for c in classes)
        order = [rng.permutation(np.flatnonzero(labels == c))[:int(
            maximum * 10 ** (-rank / (len(classes) - 1)))]
            for rank, c in enumerate(classes)]
        return rng.permutation(np.concatenate(order))
    concentration = float(protocol[len('dirichlet'):])
    slots = [[] for _ in classes]
    proportions = rng.dirichlet([concentration] * len(slots), len(classes))
    for label, partition in zip(classes, proportions):
        indices = np.flatnonzero(labels == label)
        boundaries = (np.cumsum(partition)[:-1] * len(indices)).astype(int)
        for slot, part in zip(slots, np.split(indices, boundaries)):
            slot.append(part)
    return np.concatenate([np.concatenate([slot[i] for i in rng.permutation(len(slot))])
                           for slot in slots])


class OrderedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, np.asarray(indices)
        self.video_infos = [dataset.video_infos[int(i)] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class RollInput(torch.nn.Module):
    def __init__(self, adapter, protocol):
        super().__init__()
        self.adapter, self.protocol = adapter, protocol

    def forward(self, x):
        x = roll(x, 1 if self.protocol == 'roll90' else 2)
        return self.adapter(x.flip([2]) if self.protocol.endswith('_reverse') else x)


def summarize_theory(measurements):
    correlations = [m['spearman'] for m in measurements if m['spearman'] is not None]
    ratios = [abs(m['between']) / abs(m['shared_push_D']) for m in measurements
              if abs(m['shared_push_D']) > 1e-12]
    result = dict(measured_batches=len(measurements),
                spearman_mean=float(np.mean(correlations)) if correlations else None,
                constant_rank_batches=len(measurements) - len(correlations),
                D_nonnegative_pct=float(np.mean([m['D'] >= 0 for m in measurements]) * 100),
                between_fraction_mean=float(np.mean(ratios)) if ratios else None,
                D_definition='negative parameter-gradient dot product, cross-fitted batch halves',
                decomposition_definition='analytic shared-logit push; distinct from parameter D',
                optimizer_momentum_in_D=False,
                shared_BN_statistics=True)
    if 'parameter_between' in measurements[0]:
        eligible = [m for m in measurements if abs(m['D']) > 1e-12]
        result.update(
            parameter_between_fraction_mean=float(np.mean([
                abs(m['parameter_between']) / abs(m['D']) for m in eligible])) if eligible else None,
            parameter_nonshared_fraction_mean=float(np.mean([
                abs(m['parameter_nonshared_push']) / abs(m['D']) for m in eligible])) if eligible else None,
            parameter_ratio_batches=len(eligible),
            parameter_near_zero_D_batches=len(measurements) - len(eligible),
            parameter_decomposition_max_abs_error=max(abs(m['parameter_decomposition_error'])
                                                       for m in measurements),
            parameter_decomposition_definition='D = first + between + nonshared_push; '
                                               'shared push = mean J(-grad_theta L); '
                                               'between_fraction denominator = abs(parameter D)')
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_runtime_arguments(parser)
    parser.add_argument('--mode', default='benchmark_continual')
    parser.add_argument('--methods', nargs=1, required=True)
    parser.add_argument('--offline-corruptions', nargs='+')
    parser.add_argument('--protocol', choices=LABEL_PROTOCOLS + ROLL_PROTOCOLS +
                        THEORY_PROTOCOLS + ('real',), required=True)
    args = parser.parse_args()
    cfg = build_runtime(args, args.mode)
    method, protocol = args.methods[0], args.protocol
    tags = CORRUPTIONS[:17] if protocol not in ROLL_PROTOCOLS + ('real',) else [protocol]
    if args.dry_run:
        print(json.dumps(dict(method=method, protocol=protocol, domains=len(tags),
                              batch_size=cfg.experiment.batch_size, seed=cfg.experiment.seed)))
        return
    root = Path(os.environ['PAPER_LOG_DIR']) / 'beyond' / protocol
    destination = root / method
    destination.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.experiment.seed)
    mmcv_cfg = load_mmcv_test_config(cfg)
    if protocol in THEORY_PROTOCOLS:
        model = build_model(mmcv_cfg.model, cfg.experiment.checkpoint)
        adapter = setup_theory(model, cfg, tent=method == 'Tent', exact=protocol == 'exact_gate',
                               decomposition=protocol == 'theory_decomposition')
    else:
        adapter = load_model_adapter(method, args, cfg, mmcv_cfg, method_config=args.method_config)
    log = StreamLogger(root, 'single', method, args.note or 'default', cfg.experiment.batch_size)
    test_cfg = mmcv_cfg.model.get('test_cfg', dict(average_clips='prob'))
    accuracies, counts, measurements, seconds, batches = [], [], [], 0.0, 0
    for domain, tag in enumerate(tags):
        seed = cfg.experiment.seed * 100 + domain
        set_global_seed(seed)
        if protocol in ROLL_PROTOCOLS + ('real',):
            eval_cfg = build_cfg_for_corruption(mmcv_cfg, batch_size=cfg.experiment.batch_size)
            if protocol == 'real':
                eval_cfg.data.test.split = cfg.dataset.split
            dataset, loader = build_dataset_and_loader(
                eval_cfg, seed, cache_dir=os.environ.get('EVAL_CACHE_DIR', '/tmp/protogcn-eval-cache'))
        else:
            _, dataset, loader = build_domain_loader(mmcv_cfg, cfg, None, tag, seed, True)
        if protocol in LABEL_PROTOCOLS:
            labels = [int(item['label']) for item in dataset.video_infos]
            indices = label_order(labels, protocol, seed)
            dataset = OrderedDataset(dataset, indices)
            loader = DataLoader(dataset, batch_size=cfg.experiment.batch_size,
                                num_workers=mmcv_cfg.data.workers_per_gpu, pin_memory=True)
            # Raw sampler indices make stream construction reproducible without predictions/labels leakage.
            np.save(destination / (tag + '_order.npy'), indices)
        labels = np.array([int(item['label']) for item in dataset.video_infos])
        if protocol == 'real':
            assert np.isin(labels, cfg.dataset.shared_ntu60_labels).all()
        if hasattr(adapter, 'set_stream_position'):
            adapter.set_stream_position(0, domain, tag)
        evaluated = RollInput(adapter, protocol) if protocol in ROLL_PROTOCOLS else adapter
        log.begin(adapter, 0, domain, tag)
        with GpuTimer() as timer:
            outputs = evaluate_on_loader(evaluated, loader, test_cfg)
        accuracy = float((np.asarray(outputs).argmax(1) == labels).mean() * 100)
        log.end(adapter, outputs, dataset, timer, top1=accuracy)
        accuracies.append(accuracy); counts.append(len(dataset))
        seconds += timer.seconds; batches += len(loader)
        if hasattr(adapter, 'measurements'):
            measurements.extend(dict(m, domain=tag) for m in adapter.measurements)
            adapter.measurements.clear()
        print(f'[{protocol}][{method}] {tag}: {accuracy:.2f}% ({len(dataset)} samples)', flush=True)
        del outputs, loader, dataset
        gc.collect()
    result = dict(method=method, protocol=protocol, domains=tags, accuracy=accuracies,
                  mean=float(np.mean(accuracies)), samples=counts, seed=cfg.experiment.seed,
                  batch_size=cfg.experiment.batch_size, checkpoint=cfg.experiment.checkpoint,
                  source_config=cfg.experiment.source_config, seconds=seconds,
                  ms_per_batch=seconds / batches * 1000, predict_before_update=True)
    if measurements:
        result['theory'] = summarize_theory(measurements)
        (destination / 'gradient_measurements.json').write_text(json.dumps(measurements))
    if protocol == 'real':
        result['shared_ntu60_labels'] = cfg.dataset.shared_ntu60_labels
    (destination / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
