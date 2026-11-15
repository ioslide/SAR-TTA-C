#!/usr/bin/env python3
"""Launch isolated paper experiments without modifying the paper or source data."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.runtime import ALL_CORRUPTIONS
from paper_tools.variants import ABLATION_ROWS, ORIENT_ROWS

METHODS = ['Source', 'BN', 'Tent', 'ETA', 'EATA', 'CoTTA', 'RoTTA', 'RDumb',
           'DeYO', 'NEO', 'ZeroSiam', 'SkeletonCacheUniform', 'SRT_TTA_paper']
GROUPS = ['main60', 'main120', 'long', 'ablation', 'orientation', 'sensitivity',
          'intervention', 'heldout', 'real', 'theory', 'backbone', 'cost', 'vote', 'dwell', 'plugin']
SRT_METHODS = {'SRT', 'SRT_TTA_paper', 'SREM', 'SRT_G', 'SREM_IM', 'Tent_IM'}


def jobs(args):
    group = args.group
    dataset = 'ntu120_xsub' if group == 'main120' else 'ntu60_xview'
    methods = args.methods or METHODS
    for seed in args.seeds:
        def job(method, note='', entry='run.py', extra=(), ds=dataset, label=None, env=None):
            method_config = 'Tent' if method == 'Tent_IM' else method
            work = args.output / group / ds / f'seed{seed}' / method / (label or note or 'default')
            command = [sys.executable, entry, '--mode', 'benchmark_continual',
                       '--experiment-config', f'configs/experiment/benchmark_{"ntu60_xview" if ds == "ntu60_to_pku" else ds}.yaml',
                       '--dataset-config', f'configs/dataset/{ds}.yaml',
                       '--method-config', f'configs/method/{method_config}.yaml',
                       '--methods', method, '--seed', str(seed), '--work-dir', str(work)]
            if note:
                command += ['--note', note]
            command += list(extra)
            if args.dry_run:
                command.append('--dry-run')
            variables = dict(os.environ, PAPER_LOG_DIR=str(work / 'streams'), SKIP_PKL='1')
            variables.update(env or {})
            return command, variables

        if group in ('main60', 'main120', 'long', 'cost'):
            for method in methods:
                note = ('ntu120' if group == 'main120' else 'full') if method in SRT_METHODS else ''
                extra = []
                entry = 'run.py'
                if group == 'main120':
                    extra = ['--offline-corruptions', *ALL_CORRUPTIONS[:-1]]
                if group == 'long':
                    entry = 'L-CS_run.py'
                    extra = ['--rounds', str(args.rounds), '--no-early-stop']
                if group == 'cost':
                    extra = ['--offline-corruptions', 'segment_dropout']
                yield job(method, note, entry, extra, env={'PAPER_TIMING': '1', 'PAPER_FLOPS': '1' if group == 'cost' else '0'})
        elif group in ('ablation', 'orientation', 'sensitivity', 'intervention'):
            notes = {'ablation': [n for _, n in ABLATION_ROWS],
                     'orientation': [n for _, n in ORIENT_ROWS],
                     'intervention': ['interv_all', 'interv_sa', 'interv_sr'],
                     'sensitivity': [f'sens_{key}_{v:.2f}' for key, values in
                                     [('alpha', [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3]),
                                      ('beta', [0, 1, 2, 3, 4, 5]),
                                      ('rho', [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])]
                                     for v in values]}[group]
            for note in notes:
                yield job('SRT_TTA_paper', note)
        elif group in ('heldout', 'real', 'theory'):
            protocols = {'heldout': ['roll90', 'roll180', 'roll180_reverse'],
                         'real': ['real'], 'theory': ['theory', 'theory_decomposition', 'exact_gate']}[group]
            selected = args.methods or (['Tent', 'SRT_TTA_paper'] if group == 'theory' else METHODS + ['SRT_G'])
            for protocol in protocols:
                for method in selected:
                    if group == 'theory' and protocol != 'theory' and method == 'Tent':
                        continue
                    yield job(method, 'full' if method in SRT_METHODS else '', 'paper_tools/beyond.py',
                              ['--protocol', protocol], ds='ntu60_to_pku' if group == 'real' else dataset, label=protocol)
        elif group == 'backbone':
            for ds in ['ntu60_xview_stgcnpp', 'ntu60_xview_ctrgcn', 'ntu60_xsub']:
                for method in args.methods or ['Source', 'Tent', 'SRT_TTA_paper']:
                    actual_ds = 'ntu60_xsub_skeleton_c_v2' if ds == 'ntu60_xsub' else ds
                    command, variables = job(method, 'full' if method in SRT_METHODS else '',
                                             'L-CS_run.py', ['--rounds', str(args.rounds), '--no-early-stop'], ds=ds)
                    if ds == 'ntu60_xsub':
                        command[command.index('--dataset-config') + 1] = f'configs/dataset/{actual_ds}.yaml'
                    yield command, variables
        elif group in ('vote', 'dwell'):
            if group == 'vote':
                yield job('Source', entry='paper_tools/stream_probes.py', extra=['--probe', 'vote'])
            else:
                for dwell in ['iid', '1', '2', '4', '8', '16', '32', '64', '128', '256']:
                    for method, note in [('Tent', ''), ('SRT_TTA_paper', 'full'), ('SRT_TTA_paper', 'ori_oracle')]:
                        yield job(method, note, 'paper_tools/stream_probes.py',
                                  ['--probe', 'dwell', '--dwell-batches', dwell], label=dwell + '_' + (note or 'default'))
        elif group == 'plugin':
            for method in args.methods or ['Tent', 'ETA', 'EATA', 'DeYO', 'CoTTA', 'RoTTA']:
                yield job(method, env={'PLUGIN_ATC': '1'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--methods', nargs='+', choices=METHODS + ['SRT', 'Skeleton-Cache', 'SRT_G', 'SREM', 'SREM_IM', 'Tent_IM'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--rounds', type=int, default=15, help='repetitions of the fixed domain sequence')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--dry-run', action='store_true', help='validate entries without loading data or models')
    parser.add_argument('--plan', action='store_true', help='print commands without running them')
    args = parser.parse_args()
    failed = []
    for command, variables in jobs(args):
        print(shlex.join(command), flush=True)
        if not args.plan:
            result = subprocess.run(command, cwd=ROOT, env=variables)
            if result.returncode:
                failed.append((result.returncode, command))
    for code, command in failed:
        print(f'FAILED ({code}): {shlex.join(command)}', file=sys.stderr)
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
