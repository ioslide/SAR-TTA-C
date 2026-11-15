"""Single-pass persistence sweep and frozen-source vote measurements.

Uses the existing FP32 evaluation datasets and adapters. The persistence sweep
changes only stream order; every point contains the same domain/sample multiset.
"""
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner_common import (add_runtime_arguments, average_clip, build_domain_loader,
                           build_runtime, load_mmcv_test_config, load_model_adapter,
                           set_global_seed)
from paper_tools.collect_runs import CORRUPTIONS, norm_entropy
from paper_tools.paper_logging import TIME_DIM, _entropy_from_output

DWELL_POINTS = ('iid', '1', '2', '4', '8', '16', '32', '64', '128', '256')
VOTE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def stream_order(size, domains, batch_size, dwell, seed):
    if dwell == 'iid':
        # Balanced random mixing: exactly the same examples as every chunked stream.
        return np.random.default_rng(seed).permutation(size * domains)
    length = size if dwell == 'full' else int(dwell) * batch_size
    return np.concatenate([np.arange(d * size + start, d * size + min(start + length, size))
                           for start in range(0, size, length) for d in range(domains)])


class StreamDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.size = len(datasets[0])
        assert all(len(d) == self.size for d in datasets)

    def __len__(self):
        return self.size * len(self.datasets)

    def __getitem__(self, index):
        domain, sample = divmod(int(index), self.size)
        return dict(self.datasets[domain][sample], domain=domain)


def vote_statistics(gaps):
    gaps = np.asarray(gaps, dtype=np.float64)
    mu, sigma = float(gaps.mean()), float(gaps.std())
    rows = []
    for batch_size in VOTE_BATCHES:
        count = len(gaps) // batch_size
        votes = gaps[:count * batch_size].reshape(count, batch_size).mean(1)
        rows.append(dict(B=batch_size, q=float((votes > 0).mean()), batches=count,
                         discarded=len(gaps) % batch_size,
                         bound_sg=float(-np.expm1(-batch_size * mu ** 2 / (2 * sigma ** 2))),
                         bound_range=float(-np.expm1(-batch_size * mu ** 2 / (2 * math.log(60) ** 2)))))
    return dict(mu=mu, sigma=sigma, n=len(gaps), rows=rows,
                bound_sg_kind='Empirical standard-deviation plug-in; not a certified population bound')


def measure_votes(adapter, loader):
    gaps = []
    device = next(adapter.parameters()).device
    with torch.no_grad():
        for batch in loader:
            x = batch['keypoint'].to(device, non_blocking=True)
            assert x.shape[1] == 1
            x = x[:, 0]
            gap = _entropy_from_output(adapter(x)) - _entropy_from_output(adapter(x.flip([TIME_DIM])))
            gaps.append(gap.cpu().numpy())
    return np.concatenate(gaps)


def measure_stream(adapter, loader, test_cfg, oracle=False):
    predictions, labels, domains, means, sizes = [], [], [], [], []
    device = next(adapter.parameters()).device
    for batch in loader:
        x = batch['keypoint'].to(device, non_blocking=True)
        batch_size, clips = x.shape[:2]
        x = x.reshape((batch_size * clips,) + x.shape[2:])
        if oracle:
            mask = (batch['domain'] == CORRUPTIONS.index('temporal_reverse')).to(device)
            mask = mask.repeat_interleave(clips)
            x[mask] = x[mask].flip([TIME_DIM])
        logits = adapter(x).detach()
        p = average_clip(test_cfg, logits.reshape(batch_size, clips, -1)).squeeze(1).cpu().numpy()
        predictions.append(p.argmax(1)); labels.append(batch['label'].numpy().reshape(-1))
        domains.append(batch['domain'].numpy()); means.append(p.mean(0)); sizes.append(batch_size)
        # The native SRT trace is not needed here; drain it to keep memory bounded.
        if hasattr(adapter, 'pop_paper_trace'):
            adapter.pop_paper_trace()
    pred, label, domain = map(np.concatenate, (predictions, labels, domains))
    pi = np.full(len(means[0]), 1.0 / len(means[0]))
    for row in means:
        pi = .9 * pi + .1 * row.astype(np.float64)
    correct = pred == label
    rev = domain == CORRUPTIONS.index('temporal_reverse')
    return (dict(n=len(label), mean=float(correct.mean() * 100), hpi=norm_entropy(pi),
                 reverse=float(correct[rev].mean() * 100)),
            dict(pred=pred.astype(np.int16), label=label.astype(np.int16),
                 domain=domain.astype(np.int8), bmean=np.asarray(means), bsize=np.asarray(sizes)))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_runtime_arguments(parser)
    parser.add_argument('--mode', default='benchmark_continual')
    parser.add_argument('--methods', nargs=1, required=True)
    parser.add_argument('--offline-corruptions', nargs='+', default=CORRUPTIONS)
    parser.add_argument('--probe', choices=['vote', 'dwell'], required=True)
    parser.add_argument('--dwell-batches', choices=DWELL_POINTS)
    args = parser.parse_args()
    cfg = build_runtime(args, 'benchmark_continual')
    assert args.offline_corruptions == CORRUPTIONS
    if args.dry_run:
        print(json.dumps(dict(probe=args.probe, dwell=args.dwell_batches, method=args.methods[0],
                              seed=cfg.experiment.seed, batch_size=cfg.experiment.batch_size)))
        return
    logging.basicConfig(level=logging.WARNING)
    set_global_seed(cfg.experiment.seed)
    mmcv_cfg = load_mmcv_test_config(cfg)
    root = Path(os.environ['PAPER_LOG_DIR']) / 'probes'
    metadata = dict(seed=cfg.experiment.seed, batch_size=cfg.experiment.batch_size,
                    checkpoint=cfg.experiment.checkpoint, config=cfg.experiment.source_config,
                    severity=cfg.experiment.severity)
    if args.probe == 'vote':
        assert args.methods == ['Source']
        _, _, loader = build_domain_loader(mmcv_cfg, cfg, None, 'temporal_reverse',
                                           cfg.experiment.seed * 100 + 3, True)
        adapter = load_model_adapter('Source', args, cfg, mmcv_cfg)
        gaps = measure_votes(adapter, loader)
        result = dict(metadata, **vote_statistics(gaps))
        output = root / 'vote'; output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'source_gaps.npz', gap=gaps)
        (output / 'summary.json').write_text(json.dumps(result, indent=2))
    else:
        assert args.dwell_batches is not None
        datasets = [build_domain_loader(mmcv_cfg, cfg, None, tag,
                                       cfg.experiment.seed * 100 + i, True)[1]
                    for i, tag in enumerate(CORRUPTIONS)]
        dataset = StreamDataset(datasets)
        order = stream_order(dataset.size, len(datasets), cfg.experiment.batch_size,
                             args.dwell_batches, cfg.experiment.seed)
        loader = DataLoader(dataset, batch_size=cfg.experiment.batch_size, sampler=order,
                            num_workers=mmcv_cfg.data.workers_per_gpu, pin_memory=True)
        adapter = load_model_adapter(args.methods[0], args, cfg, mmcv_cfg, args.method_config)
        adapter.episodic = False
        oracle = args.note == 'ori_oracle'
        if hasattr(adapter, 'set_stream_position'):
            adapter.set_stream_position(0, 0, 'mixed')
        test_cfg = mmcv_cfg.model.get('test_cfg', mmcv_cfg.get('test_cfg', dict(average_clips='prob')))
        result, arrays = measure_stream(adapter, loader, test_cfg, oracle)
        result.update(metadata, dwell=args.dwell_batches,
                      order='balanced random mixing' if args.dwell_batches == 'iid' else 'round-robin chunks',
                      samples_per_domain=dataset.size)
        name = 'Oracle' if oracle else ('SRT' if args.methods[0] == 'SRT_TTA_paper' else args.methods[0])
        output = root / 'dwell' / args.dwell_batches; output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / (name + '.npz'), **arrays)
        (output / (name + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
