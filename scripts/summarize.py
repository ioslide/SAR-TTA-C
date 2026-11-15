#!/usr/bin/env python3
"""Print completed primary and recurring accuracies from launcher outputs as CSV."""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.runtime import ALL_CORRUPTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=Path('outputs'))
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(['task', 'method', 'note', 'sequence_index', 'domains', 'mean_percent', 'mixed'])
    pooled = defaultdict(list)
    for index in sorted(args.output.rglob('index.jsonl')):
        if 'archive' in index.parts or 'streams' not in index.parts:
            continue
        relative = index.relative_to(args.output)
        count = 18 if any(p in ('long', 'backbone', 'ablation-long') for p in relative.parts) else 17
        domains = ALL_CORRUPTIONS[:count]
        required = set(domains)
        records = {}
        for line in index.read_text().splitlines():
            record = json.loads(line)
            records[(record['round'], record['tag'])] = record
        for sequence in sorted({r for r, _ in records}):
            if not required.issubset({t for r, t in records if r == sequence}):
                continue
            values = []
            for tag in domains:
                record = records[(sequence, tag)]
                value = record.get('top1')
                values.append(record['top1_from_labels'] if value is None else value)
            mean = statistics.mean(values)
            mixed = records.get((sequence, 'mixed_failure'), {}).get('top1_from_labels')
            task = '/'.join(relative.parts[:relative.parts.index('streams')])
            method, note = record['method'], record['note']
            writer.writerow([task, method, note, sequence + 1, count, f'{mean:.6f}', mixed])
            key = (tuple(p for p in Path(task).parts if not p.startswith('seed')), method, note, sequence, count)
            pooled[key].append(mean)
    for (task, method, note, sequence, count), values in sorted(pooled.items()):
        if len(values) > 1:
            writer.writerow(['/'.join(task) + f'/across_{len(values)}_seeds', method, note, sequence + 1, count,
                             f'{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}', ''])


if __name__ == '__main__':
    main()
