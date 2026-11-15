#!/usr/bin/env python3
"""Turn the stream logs of run.sh into the data of the paper's tables and figures.

    python paper_tools/collect_runs.py --logs paper_logs --out <paper>/data/real \
        [--ours-method SRT_TTA_paper] [--rounds 15] [--reference data/real_main_seed42.csv]

Input : $PAPER_LOG_DIR/<kind>/<method>/<note>/{index.jsonl, r??_d??_<tag>.npz}
        (kind = single | lcs<R>; backbones/<key>/... for Tab. S5)
Output: <out>/results.json (everything the paper needs, see the keys below) and
        <out>/single_runs.csv / long_runs.csv (human-readable accuracy matrices).
Only numpy is required.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_tools.variants import ABLATION_ROWS, ORIENT_ROWS  # noqa: E402

CORRUPTIONS = [
    'segment_dropout', 'segment_shuffle', 'temporal_delay', 'temporal_reverse',
    'camera_viewpoint', 'nonrigid_joint_drift', 'coordinate_quantization', 'skeleton_mirror',
    'persistent_joint_missing', 'joint_dropout_burst', 'identity_switch',
    'left_right_asymmetry', 'joint_topology_swap',
    'temporal_correlated_noise', 'outlier_frames', 'depth_axis_noise', 'high_frequency_jitter',
    'mixed_failure',
]
REVERSE = 'temporal_reverse'
RHO_PI = 0.9                       # same momentum as the running marginal of SRT
# NTU-60 reversal-aliased pairs (0-based labels): A8/A9, A14/A15, A16/A17, A18/A19, A20/A21, A59/A60
PAIRS = [('SitStand', 7, 8), ('Jacket', 13, 14), ('Shoe', 15, 16), ('Glasses', 17, 18),
         ('Hat', 19, 20), ('Walk', 58, 59)]
DEFAULT_KEYS = {'Source': 'Source', 'SRT': 'SRT', 'Skeleton-Cache': 'Skeleton-Cache', 'SkeletonCacheUniform': 'Skeleton-Cache', 'BN': 'BN', 'Tent': 'TENT', 'TENT': 'TENT',
                'ETA': 'ETA', 'EATA': 'EATA', 'DeYO': 'DeYO', 'CoTTA': 'CoTTA', 'RoTTA': 'RoTTA',
                'SRT_TTA_paper': 'SRT',
                'Ours': 'SRT'}


# ====================================================================== loading
class Run:
    """All logged domains of one (kind, method, note) run."""

    def __init__(self, path, kind, method, note):
        self.path, self.kind, self.method, self.note = path, kind, method, note
        self.records = {}
        index = path / 'index.jsonl'
        if index.exists():
            for line in index.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[(rec['round'], rec['tag'])] = rec   # the last record wins
        self.rounds = 1 if kind == 'single' else int(kind[3:])
        self._cache = {}

    def npz(self, rnd, tag):
        key = (rnd, tag)
        if key not in self._cache:
            rec = self.records.get(key)
            if rec is None or not (self.path / rec['file']).exists():
                self._cache[key] = None
            else:
                with np.load(self.path / rec['file']) as z:
                    self._cache[key] = {k: z[k] for k in z.files}
        return self._cache[key]

    def drop_cache(self):
        self._cache = {}

    def acc(self, rnd, tag):
        rec = self.records.get((rnd, tag))
        if rec is None:
            return None
        return rec['top1'] if rec.get('top1') is not None else rec.get('top1_from_labels')

    def round_acc(self, rnd):
        return [self.acc(rnd, t) for t in CORRUPTIONS]

    def complete_rounds(self):
        n = 0
        while n < self.rounds and all(a is not None for a in self.round_acc(n)):
            n += 1
        return n


def load_runs(root):
    runs = {}
    root = Path(root)
    if not root.exists():
        return runs
    for kind_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        kind = kind_dir.name
        if kind != 'single' and not (kind.startswith('lcs') and kind[3:].isdigit()):
            continue
        for method_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            for note_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                run = Run(note_dir, kind, method_dir.name, note_dir.name)
                if run.records:
                    runs[(kind, method_dir.name, note_dir.name)] = run
    return runs


# ====================================================================== helpers
def mean_or_none(values):
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def norm_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def iterate_stream(run, max_rounds=None):
    """Yield (round, domain index, tag, npz) in stream order for every logged domain."""
    rounds = run.rounds if max_rounds is None else min(run.rounds, max_rounds)
    for rnd in range(rounds):
        for d, tag in enumerate(CORRUPTIONS):
            z = run.npz(rnd, tag)
            if z is None:
                return
            yield rnd, d, tag, z


def marginal_trajectory(run, max_rounds=None, per_batch=False):
    """EMA of the batch-mean prediction (rho = 0.9), continued across domains and rounds."""
    pi, ends, trace = None, [], []
    for rnd, d, tag, z in iterate_stream(run, max_rounds):
        bmean = z['bmean'].astype(np.float64)
        if pi is None:
            pi = np.full(bmean.shape[1], 1.0 / bmean.shape[1])
        for row in bmean:
            pi = RHO_PI * pi + (1 - RHO_PI) * row
            if per_batch:
                trace.append(norm_entropy(pi))
        if d == len(CORRUPTIONS) - 1:
            ends.append((norm_entropy(pi), pi.copy()))
    return ends, trace


def batch_accuracy(z):
    if 'label' not in z:
        return None
    correct = (z['pred'].astype(np.int64) == z['label'].astype(np.int64)).astype(np.float64)
    sizes = z['bsize'].astype(np.int64)
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return np.array([correct[a:b].mean() * 100.0 for a, b in zip(edges[:-1], edges[1:])])


def smooth(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, values[0]), values, np.full(window - 1 - window // 2, values[-1])])
    return np.convolve(padded, kernel, mode='valid')


def downsample(x, ys, n=400):
    if len(x) <= n:
        return list(map(float, x)), [list(map(float, y)) for y in ys]
    idx = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
    return [float(x[i]) for i in idx], [[float(y[i]) for i in idx] for y in ys]


def stream_curve(run, rounds, with_hpi=False):
    """Per-batch accuracy (and marginal entropy) over the first `rounds` windows, x in processed mini-batches."""
    accs, sizes = [], []
    for rnd, d, tag, z in iterate_stream(run, rounds):
        a = batch_accuracy(z)
        if a is None:
            return None
        accs.append(a)
        sizes.append(len(a))
    if not accs or len(accs) < rounds * len(CORRUPTIONS):
        return None
    acc = np.concatenate(accs)
    per_round = sum(sizes[:len(CORRUPTIONS)])
    x = np.arange(1, len(acc) + 1)
    out = {'x': x, 'acc': smooth(acc, max(1, per_round // 40))}
    if with_hpi:
        _, hpi = marginal_trajectory(run, rounds, per_batch=True)
        out['hpi'] = np.asarray(hpi)
    return out


def trace_latencies(run, rnd):
    """Batches until the orientation state crosses zero after entering / leaving the reversed domain."""
    rev_idx = CORRUPTIONS.index(REVERSE)
    z_in = run.npz(rnd, REVERSE)
    z_out = run.npz(rnd, CORRUPTIONS[rev_idx + 1]) if rev_idx + 1 < len(CORRUPTIONS) else None
    entry = exit_ = None
    if z_in is not None and 'tr_s' in z_in:
        pos = np.nonzero(z_in['tr_s'] > 0)[0]
        entry = int(pos[0]) + 1 if len(pos) else None
    if z_out is not None and 'tr_s' in z_out:
        pos = np.nonzero(z_out['tr_s'] <= 0)[0]
        exit_ = int(pos[0]) + 1 if len(pos) else None
    return entry, exit_


def misorientation(run, rnd=0):
    """Wrongly oriented batches (batch rules) / samples (per-sample rule) outside the reversed domain."""
    batches, weighted, total, seen = 0, 0.0, 0, False
    for tag in CORRUPTIONS[:-1]:
        z = run.npz(rnd, tag)
        if z is None or 'tr_o' not in z:
            continue
        seen = True
        if tag == REVERSE:
            continue
        o = z['tr_o'].astype(np.float64)
        sizes = z['bsize'].astype(np.float64)[:len(o)]
        batches += int((o > 0.5).sum())
        weighted += float((o * sizes).sum())
        total += float(sizes.sum())
    if not seen:
        return None, None
    return batches, (100.0 * weighted / total if total else None)


def pair_swap(z):
    if z is None or 'label' not in z:
        return None
    pred, label = z['pred'].astype(np.int64), z['label'].astype(np.int64)
    out = []
    for _, a, b in PAIRS:
        mask = (label == a) | (label == b)
        if mask.sum() == 0:
            out.append(None)
            continue
        partner = np.where(label[mask] == a, b, a)
        out.append(float((pred[mask] == partner).mean() * 100.0))
    return out


# ====================================================================== main collection
def summarize_single(run, rnd=0):
    acc = run.round_acc(rnd)
    recs = [run.records.get((rnd, t)) for t in CORRUPTIONS]
    recs = [r for r in recs if r]
    fwd_s = sum(r.get('fwd_seconds') or 0.0 for r in recs)
    fwd_n = sum(r.get('fwd_calls') or 0 for r in recs)
    secs = sum(r.get('seconds') or 0.0 for r in recs)
    nb = sum(r.get('n_batches') or 0 for r in recs)
    mems = [r.get('peak_mem_gb') for r in recs if r.get('peak_mem_gb') not in (None,) and not
            (isinstance(r.get('peak_mem_gb'), float) and math.isnan(r['peak_mem_gb']))]
    mis_b, mis_f = misorientation(run, rnd)
    entry, exit_ = trace_latencies(run, rnd)
    ends, _ = marginal_trajectory(run, rnd + 1)
    return {
        'acc': acc, 'mean': mean_or_none(acc[:-1]),
        'ms_per_batch': (1000.0 * fwd_s / fwd_n) if fwd_n else ((1000.0 * secs / nb) if nb else None),
        'timed_forward': bool(fwd_n),
        'peak_mem_gb': max(mems) if mems else None,
        'misorient_batches': mis_b, 'misorient_sample_pct': mis_f,
        'entry_latency': entry, 'exit_latency': exit_,
        'hpi_end': ends[rnd][0] if len(ends) > rnd else None,
        'source': f'{run.kind}/{run.method}/{run.note}' + ('' if run.kind == 'single' else f' (round {rnd})'),
    }


def collect(args):
    keys = dict(DEFAULT_KEYS)
    keys[args.ours_method] = 'SRT'
    for item in args.method_map:
        name, key = item.split('=', 1)
        keys[name] = key
    paper_key = lambda m: keys.get(m, m)  # noqa: E731

    runs = load_runs(args.logs)
    print(f'[collect] {len(runs)} logged runs in {args.logs}')
    res = {'meta': {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'logs': str(args.logs),
                    'ours_method': args.ours_method, 'domains': CORRUPTIONS, 'rounds': args.rounds},
           'single': {}, 'long': {}, 'curves': {}, 'backbones': {}}

    single_runs, long_runs = {}, {}
    for (kind, method, note), run in runs.items():
        k = f'{paper_key(method)}|{note}'
        if kind == 'single':
            single_runs[k] = run
        else:
            prev = long_runs.get(k)
            if prev is None or (min(run.complete_rounds(), args.rounds), run.rounds == args.rounds) > (
                    min(prev.complete_rounds(), args.rounds), prev.rounds == args.rounds):
                long_runs[k] = run

    for run in runs.values():                       # domain / batch size for the paper's arithmetic
        rec = next(iter(run.records.values()))
        res['meta'].setdefault('domain_size', rec.get('n'))
        res['meta'].setdefault('batch_size', rec.get('batch_size'))
    for k, run in single_runs.items():
        res['single'][k] = summarize_single(run)
        run.drop_cache()
    for k, run in long_runs.items():
        n_done = min(run.complete_rounds(), args.rounds)
        if n_done == 0:
            continue
        accs = [run.round_acc(r) for r in range(n_done)]
        ends, _ = marginal_trajectory(run, n_done)
        res['long'][k] = {'rounds': min(run.rounds, args.rounds), 'complete_rounds': n_done, 'acc': accs,
                          'round_mean': [mean_or_none(a) for a in accs],
                          'hpi_round_end': [e[0] for e in ends]}
        if k not in res['single']:                  # round 0 of a recurring run = a single pass
            entry = summarize_single(run, 0)
            entry['from_long'] = True
            res['single'][k] = entry
        run.drop_cache()

    def single_run(k):
        return single_runs.get(k) or long_runs.get(k)

    curves = res['curves']
    # Reversal-type accuracies use the class groups listed in the paper's appendix.
    invariant = {2, 3, 9, 22, 25, 29, 33, 35, 48}
    aliased = {c for _, a, b in PAIRS for c in (a, b)}
    groups = {'I': invariant, 'A': aliased, 'D': set(range(60)) - invariant - aliased}
    reversal_types = {}
    for key in ('Source|default', 'TENT|default', 'CoTTA|default', 'SRT|full',
                'SRT|ori_per_sample', 'SRT|ori_oracle'):
        run = single_run(key)
        if run is None or not run.complete_rounds():
            continue
        z = run.npz(0, REVERSE)
        if z is not None and 'label' in z:
            correct = z['pred'] == z['label']
            reversal_types[key] = {
                g: float(correct[np.isin(z['label'], list(ids))].mean() * 100.0)
                for g, ids in groups.items()}
            reversal_types[key]['all'] = float(correct.mean() * 100.0)
        run.drop_cache()
    curves['reversal_types'] = reversal_types
    # ---- Fig. 2(a): marginal entropy and accuracy of TENT over two rounds
    run = long_runs.get('TENT|default')
    if run is not None and run.rounds >= 2:
        c = stream_curve(run, 2, with_hpi=True)
        if c is not None:
            peak_i = int(np.argmax(c['acc'][:len(c['acc']) // 2]))
            peak = float(c['acc'][peak_i])
            onset = next((float(c['x'][i]) for i, h in enumerate(c['hpi']) if h < 0.8), None)
            col = next((float(c['x'][i]) for i in range(peak_i, len(c['acc'])) if c['acc'][i] < peak / 2), None)
            x, (hpi, acc) = downsample(c['x'], [c['hpi'], c['acc']])
            curves['sink'] = {'x': x, 'hpi': hpi, 'acc': acc, 'peak': peak, 'onset': onset, 'collapse': col}
        run.drop_cache()
    # ---- Fig. 2(b): confidence-matched intervention
    inter = {}
    for name, cands in (('all', ['SRT|interv_all', 'SRT|ablation_vanilla']),
                        ('SA', ['SRT|interv_sa']), ('SR', ['SRT|interv_sr'])):
        for k in cands:
            run = long_runs.get(k)
            if run is not None and run.rounds >= args.interv_rounds:
                c = stream_curve(run, args.interv_rounds)
                run.drop_cache()
                if c is not None:
                    inter[name] = c
                    break
    if len(inter) == 3:
        n = min(len(v['x']) for v in inter.values())
        x, ys = downsample(inter['all']['x'][:n], [inter[k]['acc'][:n] for k in ('all', 'SA', 'SR')])
        curves['intervention'] = {'x': x, 'all': ys[0], 'SA': ys[1], 'SR': ys[2], 'rounds': args.interv_rounds}
    # ---- Fig. 2(c): entropy gap of the source model
    run = single_runs.get('Source|diag_source_gap')
    if run is not None:
        gaps = []
        for tag in CORRUPTIONS:
            z = run.npz(0, tag)
            if z is None or 'tr_h_x' not in z:
                gaps = None
                break
            w = z['bsize'].astype(np.float64)[:len(z['tr_h_x'])]
            gaps.append(float(((z['tr_h_x'] - z['tr_h_r']) * w).sum() / w.sum()))
        if gaps:
            curves['entropy_gap'] = {'gap': gaps}
        run.drop_cache()
    # ---- Fig. 2(d): pair-swap rate on the reversed domain + Fig. 1(a) source confidence
    pairs = {}
    for name, k in (('Source', 'Source|default'), ('TENT', 'TENT|default'), ('CoTTA', 'CoTTA|default'),
                    ('SRT', 'SRT|full')):
        run = single_run(k)
        if run is not None:
            sw = pair_swap(run.npz(0, REVERSE))
            if sw is not None and all(v is not None for v in sw):
                pairs[name] = sw
            if name == 'Source':
                z = run.npz(0, REVERSE)
                if z is not None and 'label' in z:
                    m = (z['label'] == 7) & (z['pred'] == 8)
                    if m.any():
                        curves['teaser_conf'] = float(np.median(z['conf'][m].astype(np.float64)))
            run.drop_cache()
    if pairs:
        curves['pairs'] = dict(names=[p[0] for p in PAIRS], **pairs)
    # ---- Fig. 1(b): class marginal after one round
    tm = {}
    for name, k in (('TENT', 'TENT|default'), ('SRT', 'SRT|full')):
        run = single_run(k)
        if run is not None:
            ends, _ = marginal_trajectory(run, 1)
            run.drop_cache()
            if ends:
                pi = np.sort(ends[0][1])[::-1]
                tm[name] = {'marginal': [float(v) for v in pi], 'hpi': ends[0][0],
                            'top3': float(pi[:3].sum() * 100.0)}
    if tm:
        curves['teaser_marginal'] = tm
    # ---- Fig. 1(b): single-pass orientation and marginal entropy, on one batch grid
    tent, ours = single_runs.get('TENT|default'), single_runs.get('SRT|full')
    if tent is not None and ours is not None and tent.complete_rounds() and ours.complete_rounds():
        _, h_tent = marginal_trajectory(tent, 1, per_batch=True)
        _, h_ours = marginal_trajectory(ours, 1, per_batch=True)
        xs, states = [], []
        for _, d, _, z in iterate_stream(ours, 1):
            s = z['tr_s']
            xs.extend(d + np.arange(len(s)) / len(s))
            states.extend(s)
        x, (s, ht, ha) = downsample(xs, [states, h_tent, h_ours], n=900)
        curves['stream_overview'] = {'x': x, 's': s, 'hpiT': ht, 'hpiA': ha}
        tent.drop_cache()
        ours.drop_cache()
    # ---- Fig. 4(a): marginal entropy at the end of every round
    me = {}
    for name in ('TENT', 'DeYO', 'CoTTA', 'EATA', 'ETA', 'SRT'):
        k = f'{name}|full' if name == 'SRT' else f'{name}|default'
        if k in res['long']:
            me[name] = res['long'][k]['hpi_round_end']
    if me:
        curves['marginal_entropy_rounds'] = me
    # ---- Fig. 4(b): orientation state of SRT in round 2 (first six domains)
    run = long_runs.get('SRT|full')
    rnd = 1 if run is not None and run.rounds > 1 else 0
    run = run or single_runs.get('SRT|full')
    if run is not None:
        xs, ss = [], []
        for d, tag in enumerate(CORRUPTIONS[:6]):
            z = run.npz(rnd, tag)
            if z is None or 'tr_s' not in z:
                xs = None
                break
            s = z['tr_s']
            xs += list(d + np.arange(len(s)) / len(s))
            ss += list(s)
        if xs:
            entry, exit_ = trace_latencies(run, rnd)
            x, (s,) = downsample(np.asarray(xs), [np.asarray(ss)], n=1200)
            curves['orientation_trace'] = {'x': x, 's': s, 'round': rnd + 1,
                                           'entry_latency': entry, 'exit_latency': exit_}
        run.drop_cache()

    # ---- Tab. S5: other backbones / splits
    bb_root = Path(args.logs) / 'backbones'
    if bb_root.exists():
        for key_dir in sorted(p for p in bb_root.iterdir() if p.is_dir()):
            out = {}
            for (kind, method, note), run in load_runs(key_dir).items():
                n_done = min(run.complete_rounds(), args.rounds)
                if n_done == 0:
                    continue
                r0 = run.round_acc(0)
                final = mean_or_none(run.round_acc(args.rounds - 1)) if n_done >= args.rounds else None
                if method == 'Source':
                    final = mean_or_none(r0)
                out[paper_key(method)] = {'mean': mean_or_none(r0[:-1]), 'r15': final,
                                          'rev': r0[CORRUPTIONS.index(REVERSE)],
                                          'rounds': n_done}
            if out:
                res['backbones'][key_dir.name] = out

    # ---- reference table (the reported main results), for comparison only
    if args.reference and Path(args.reference).exists():
        ref = {}
        with open(args.reference) as fh:
            for row in csv.DictReader(fh):
                ref[paper_key(row['Method'])] = float(row['avg'])
        res['meta']['reference_avg'] = ref
        for k, entry in res['single'].items():
            name, note = k.split('|', 1)
            if note in ('default', 'full') and name in ref and entry['mean'] is not None:
                diff = entry['mean'] - ref[name]
                flag = '' if abs(diff) < 0.3 else '   <-- differs from the reported table'
                print(f'[collect] {k:28s} mean {entry["mean"]:6.2f}  reported {ref[name]:6.2f}{flag}')
    return res


def write_outputs(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'results.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    with open(out / 'single_runs.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['key'] + CORRUPTIONS + ['avg', 'source'])
        for k, e in sorted(res['single'].items()):
            w.writerow([k] + ['' if a is None else f'{a:.2f}' for a in e['acc']]
                       + ['' if e['mean'] is None else f'{e["mean"]:.2f}', e['source']])
    with open(out / 'long_runs.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        n = max([len(e['round_mean']) for e in res['long'].values()] or [0])
        w.writerow(['key'] + [f'round_{i}' for i in range(n)] + ['final_marginal_entropy'])
        for k, e in sorted(res['long'].items()):
            means = ['' if m is None else f'{m:.2f}' for m in e['round_mean']]
            w.writerow([k] + means + [''] * (n - len(means)) + [f'{e["hpi_round_end"][-1]:.3f}'])
    print(f'[collect] wrote {out / "results.json"}: {len(res["single"])} single, {len(res["long"])} recurring, '
          f'curves: {sorted(res["curves"])}, backbones: {sorted(res["backbones"])}')
    expected = [n for _, n in ABLATION_ROWS] + [n for _, n in ORIENT_ROWS]
    missing = sorted({f'SRT|{n}' for n in expected} - set(res['single']))
    if missing:
        print(f'[collect] not yet available (placeholders stay): {missing}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', required=True, help='PAPER_LOG_DIR of run.sh')
    ap.add_argument('--out', required=True, help='output directory, normally <paper>/data/real')
    ap.add_argument('--ours-method', default='SRT_TTA_paper')
    ap.add_argument('--rounds', type=int, default=15, help='maximum recurring rounds included in paper results')
    ap.add_argument('--interv-rounds', type=int, default=2)
    ap.add_argument('--reference', help='CSV of the reported main table, only used for a consistency check')
    ap.add_argument('--method-map', nargs='*', default=[], metavar='RUN=KEY',
                    help='extra method-name -> paper-key mappings, e.g. MyTent=TENT')
    args = ap.parse_args()
    write_outputs(collect(args), args.out)


if __name__ == '__main__':
    main()
