"""Catalogue explicit inverse-action pairs and audit logged reversal predictions.

Pair membership is a semantic annotation of the official action names. Partner
prediction rates are empirical diagnostics, not proof of distributional aliasing.
No model is trained or evaluated by this CPU-only analysis.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_tools.collect_runs import CORRUPTIONS, PAIRS, REVERSE, load_runs

LABEL_SOURCE = 'https://github.com/kennymckormick/pyskl/blob/main/tools/data/label_map/nturgbd_120.txt'
ADDITIONAL = (
    (61, 'put on headphone', 62, 'take off headphone'),
    (87, 'put on bag', 88, 'take off bag'),
    (89, 'put something into a bag', 90, 'take something out of a bag'),
)


def analyse(logs, labels_path, output):
    labels = labels_path.read_text().splitlines()
    assert len(labels) == 120
    extra = []
    for a, name_a, b, name_b in ADDITIONAL:
        assert labels[a - 1] == name_a and labels[b - 1] == name_b
        extra.append(dict(action_ids=[a, b], names=[name_a, name_b]))
    previous = [dict(action_ids=[a + 1, b + 1], names=[labels[a], labels[b]])
                for _, a, b in PAIRS]
    pairs = previous + extra
    measurements = []
    runs = load_runs(logs)
    for (kind, method, note), run in runs.items():
        if kind != 'single' or (0, REVERSE) not in run.records:
            continue
        record = run.records[0, REVERSE]
        assert record['num_classes'] == 120
        with np.load(run.path / record['file']) as data:
            pred, truth = data['pred'].astype(int), data['label'].astype(int)
        assert len(truth) == record['n'] and np.all((truth >= 0) & (truth < 120))
        for pair in pairs:
            a, b = pair['action_ids']
            # Include both the pooled pair and each direction; IDs in the logs are zero-based.
            for action in [None, a, b]:
                mask = np.isin(truth, [a - 1, b - 1]) if action is None else truth == action - 1
                count = int(mask.sum())
                assert count > 0
                partner = np.where(truth[mask] == a - 1, b - 1, a - 1)
                correct = int((pred[mask] == truth[mask]).sum())
                swapped = int((pred[mask] == partner).sum())
                measurements.append(dict(method=method, note=note, a=a, b=b,
                                         group='NTU120_added' if a > 60 else 'NTU60_existing',
                                         direction='pooled' if action is None else f'A{action}',
                                         n=count, correct=correct, partner_predictions=swapped,
                                         accuracy=100.0 * correct / count,
                                         partner_rate=100.0 * swapped / count,
                                         full_stream_complete=all(run.acc(0, t) is not None for t in CORRUPTIONS[:-1])))
    assert measurements, 'No completed NTU120 time-reversal domain is available'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(label_source=LABEL_SOURCE, label_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                  label_ids='NTU action IDs are 1-based; logged class labels are 0-based',
                  annotation_basis='Explicit inverse-action names; class-conditional distribution equality is not established',
                  existing_pairs=previous, additional_pairs=extra, additional_pair_count=len(extra),
                  additional_action_count=len({x for p in extra for x in p['action_ids']}),
                  total_catalogued_pairs=len(pairs), measurements=measurements)
    (output / 'reversal_pairs.json').write_text(json.dumps(result, indent=2))
    with (output / 'reversal_pair_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(measurements[0]))
        writer.writeheader(); writer.writerows(measurements)
    partners = {a: b for p in pairs for a, b in (p['action_ids'], p['action_ids'][::-1])}
    with (output / 'action_catalogue.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['action_id', 'action_name', 'inverse_candidate_id', 'annotation'])
        for i, name in enumerate(labels, 1):
            writer.writerow([i, name, partners.get(i, ''), 'explicit_inverse_pair' if i in partners else 'not_annotated'])
    report = ['# NTU120 reversal-pair audit', '',
              f'Additional explicitly inverse-named pairs: **{len(extra)}**; actions: **{result["additional_action_count"]}**.',
              '', 'These are semantic candidate pairs. Prediction statistics below do not establish exact distributional aliasing.', '',
              '| Action IDs | Action pair |', '|---|---|']
    report += [f'| A{p["action_ids"][0]} / A{p["action_ids"][1]} | {p["names"][0]} / {p["names"][1]} |' for p in extra]
    report += ['', '| Method | Pair | Samples | Accuracy (%) | Partner rate (%) |', '|---|---|---:|---:|---:|']
    report += [f'| {m["method"]} ({m["note"]}) | A{m["a"]}/A{m["b"]} | {m["n"]} | {m["accuracy"]:.2f} | {m["partner_rate"]:.2f} |'
               for m in measurements if m['group'] == 'NTU120_added' and m['direction'] == 'pooled']
    (output / 'summary.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(additional_pairs=len(extra), total_catalogued_pairs=len(pairs),
                          measured_runs=len({(m['method'], m['note']) for m in measurements}), output=str(output))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=Path('ntu120_runs/paper_logs'))
    parser.add_argument('--labels', type=Path, default=Path('paper_tools/ntu120_action_labels.txt'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyse(args.logs, args.labels, args.out)
