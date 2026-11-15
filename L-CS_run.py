import argparse
import gc
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

from core.io import (
    dump_json,
    read_top1_matrix_csv,
    top1_matrix_completed,
    update_top1_matrix_csv,
)
from core.runtime import ALL_CORRUPTIONS
from runner_common import (
    add_runtime_arguments,
    build_domain_loader,
    build_runtime,
    evaluate_on_loader,
    extract_top1,
    initialize_run,
    load_model_adapter,
    load_mmcv_test_config,
    result_dataset_name,
    result_csv_path,
    set_global_seed,
    write_dry_run,
)
# [paper] stream logging, adapter wrappers, timing (see paper_tools/paper_logging.py)
from paper_tools.paper_logging import (
    GpuTimer,
    StreamLogger,
    maybe_wrap_adapter,
    paper_csv_path,
    pkl_name,
    skip_pkl,
)


class _QuietDatasetLogger(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        return ('videos remain after valid thresholding' not in message
                and not message.startswith('Using mmap evaluation cache:'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run continual adaptation across an ordered corruption sequence',
        allow_abbrev=False,
    )
    parser.add_argument('--mode', required=True, choices=['benchmark_continual'])
    add_runtime_arguments(parser)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--rounds', type=int, help='number of continual rounds (default: 15)')
    parser.add_argument('--no-early-stop', action='store_true', help='complete all requested rounds even after accuracy collapse')
    parser.add_argument('--ann-file', help='evaluate one annotation file as a continuous run')
    parser.add_argument(
        '--offline-corruptions',
        nargs='+',
        help='ordered offline corruption categories; defaults to ALL_CORRUPTIONS',
    )
    return parser.parse_args()


def build_domain_order(args):
    if args.offline_corruptions:
        return [(None, name) for name in args.offline_corruptions], True
    if args.ann_file:
        return [(None, Path(args.ann_file).stem.lower())], False
    return [(None, name) for name in ALL_CORRUPTIONS], True


def run_benchmark_continual(args, cfg):
    if args.ann_file and args.offline_corruptions:
        raise ValueError('--ann-file cannot be combined with --offline-corruptions')

    rounds = args.rounds if args.rounds is not None else getattr(cfg.experiment, 'rounds', 15)
    note = str(getattr(cfg.experiment, 'note', '') or '')
    if rounds < 1:
        raise ValueError('--rounds must be at least 1')
    set_global_seed(cfg.experiment.seed)
    order, offline = build_domain_order(args)
    methods = list(args.methods or getattr(cfg.experiment, 'methods', []))
    if not methods:
        methods = [cfg.method.name]
    if methods == ['Source']:
        rounds = 1
    cfg.experiment.rounds = rounds
    cfg.experiment.methods = methods

    result_severity = cfg.experiment.severity
    if offline:
        result_severity = getattr(
            cfg.dataset,
            'offline_corruption_severity',
            result_severity,
        )

    run_name, paths, logger = initialize_run(args, cfg)
    logging.getLogger('protogcn').addFilter(_QuietDatasetLogger())
    summary_path = result_csv_path(args, cfg)
    summary_path = summary_path.with_name(f'L-CS_{summary_path.name}')
    summary_path = paper_csv_path(summary_path)  # [paper] PAPER_CSV_TAG
    progress_path = paths['progress'] / f'{run_name}.json'
    order_path = paths['orders'] / f'{run_name}.json'
    dump_json(order_path, {'order': order, 'seed': cfg.experiment.seed, 'rounds': rounds})

    full_sequence = len(order) > 1 or rounds > 1
    round_columns = [f'round_{index}' for index in range(rounds)]

    if args.dry_run:
        write_dry_run(
            run_name,
            paths,
            logger,
            {
                'mode': cfg.experiment.mode,
                'methods': methods,
                'order': order,
                'rounds': rounds,
                'early_stop': not args.no_early_stop,
                'checkpoint': cfg.experiment.checkpoint,
                'source_config': cfg.experiment.source_config,
                'dataset_ann_file': args.ann_file or getattr(cfg.dataset, 'ann_file', None),
                'conda_env': cfg.experiment.conda_env,
            },
        )
        return

    mmcv_cfg = load_mmcv_test_config(cfg)
    if args.ann_file:
        mmcv_cfg.data.test.ann_file = args.ann_file
    test_cfg = mmcv_cfg.model.get(
        'test_cfg',
        mmcv_cfg.get('test_cfg', dict(average_clips='prob')),
    )

    for method_name in methods:
        adapter = load_model_adapter(
            method_name, args, cfg, mmcv_cfg, method_config=args.method_config,
        )
        if method_name.lower() in ('smi', 'pni'):
            effective = {
                'adapter': adapter.adapter,
                'source_stats_loaded': adapter.stats is not None,
            }
            if method_name.lower() == 'pni':
                effective.update(lam=adapter.lam, gamma=adapter.gamma,
                                 max_parallel=adapter.max_parallel)
            dump_json(paths['orders'] / f'{method_name}_effective_config.json', effective)
            logger.info('[%s] effective config: %s', method_name, effective)
        if full_sequence and getattr(adapter, 'episodic', False):
            logger.warning(
                '[%s] disabling episodic reset for continuous cross-corruption adaptation',
                method_name,
            )
            adapter.episodic = False
        # [paper] optional wrappers and per-domain stream log (one log for the whole stream)
        adapter = maybe_wrap_adapter(adapter, logger)
        stream_log = StreamLogger.from_env(f'lcs{rounds}', method_name, note,
                                           cfg.experiment.batch_size)

        low_accuracy_rounds = 0
        for round_index in range(rounds):
            round_path = summary_path.with_name(
                f'{summary_path.stem}_round_{round_index}{summary_path.suffix}'
            )
            rows = read_top1_matrix_csv(round_path)
            completed = set() if full_sequence else top1_matrix_completed(rows)
            logger.info('[%s] starting round %s/%s', method_name, round_index + 1, rounds)
            for index, (corruption_type, corruption_tag) in enumerate(order):
                key = (method_name, note, corruption_tag)
                if not full_sequence and key in completed:
                    logger.info('skip existing %s', key)
                    continue

                set_global_seed(cfg.experiment.seed * 100 + index)
                eval_cfg, dataset, data_loader = build_domain_loader(
                    mmcv_cfg,
                    cfg,
                    corruption_type,
                    corruption_tag,
                    cfg.experiment.seed * 100 + index,
                    offline,
                )
                if hasattr(adapter, 'calibrate'):
                    adapter.calibrate(data_loader)
                trace_start = len(adapter.trace) if hasattr(adapter, 'trace') else None
                stream_log.begin(adapter, round_index, index, corruption_tag)  # [paper]
                with GpuTimer() as timer:  # [paper]
                    outputs = evaluate_on_loader(adapter, data_loader, test_cfg)
                if trace_start is not None:
                    dump_json(
                        paths['orders'] / f'{method_name}_round_{round_index}_{corruption_tag}_operators.json',
                        {
                            'method': method_name,
                            'note': note,
                            'round': round_index,
                            'corruption': corruption_tag,
                            'operators': adapter.trace[trace_start:],
                        },
                    )
                if not skip_pkl(round_index):  # [paper] SKIP_PKL=1 / rest
                    out_file = paths['pkl'] / pkl_name(
                        method_name, note, corruption_tag, cfg.experiment.severity, round_index,
                    )
                    dataset.dump_results(outputs, out=str(out_file))
                with redirect_stdout(io.StringIO()):
                    metrics = dataset.evaluate(outputs, metrics=['top_k_accuracy'])
                top1_acc = extract_top1(metrics)
                stream_log.end(adapter, outputs, dataset, timer, top1=top1_acc * 100.0)  # [paper]
                completed.add(key)
                rows = update_top1_matrix_csv(
                    round_path,
                    method_name,
                    corruption_tag,
                    top1_acc,
                    [tag for _, tag in order],
                    seed=cfg.experiment.seed,
                    dataset_name=result_dataset_name(args, cfg),
                    severity=result_severity,
                    batch_size=cfg.experiment.batch_size,
                    note=note,
                )
                dump_json(progress_path, {'round': round_index, 'results': rows})
                logger.info(
                    '[%s][round_%s] %s s%s: top1=%.2f%%',
                    method_name,
                    round_index,
                    corruption_tag,
                    cfg.experiment.severity,
                    top1_acc * 100.0,
                )
                del outputs, metrics, data_loader, dataset, eval_cfg
                gc.collect()

            rows = read_top1_matrix_csv(round_path)
            result = next(
                (row for row in rows
                 if row.get('Method') == method_name and row.get('note', '') == note),
                {},
            )
            logger.info('[%s][round_%s] accuracy summary (note=%s)',
                        method_name, round_index, note or '-')
            logger.info('%-32s %10s', 'Corruption', 'Top-1 (%)')
            logger.info('%-32s %10s', '-' * 32, '-' * 10)
            accuracies = []
            for _, corruption_tag in order:
                value = result.get(corruption_tag, '')
                if value in ('', None):
                    continue
                accuracy = float(value)
                accuracies.append(accuracy)
                logger.info('%-32s %9.2f%%', corruption_tag, accuracy)
            round_accuracy = sum(accuracies) / len(accuracies)
            logger.info('%-32s %9.2f%%', 'Average', round_accuracy)
            rows = update_top1_matrix_csv(
                summary_path,
                method_name,
                f'round_{round_index}',
                round_accuracy / 100.0,
                round_columns,
                seed=cfg.experiment.seed,
                dataset_name=result_dataset_name(args, cfg),
                severity=result_severity,
                batch_size=cfg.experiment.batch_size,
                note=note,
            )
            dump_json(progress_path, {'round': round_index, 'results': rows})
            low_accuracy_rounds = low_accuracy_rounds + 1 if round_accuracy < 5.0 else 0
            if not args.no_early_stop and low_accuracy_rounds > 2:
                logger.info(
                    '[%s] early stopping after %s rounds: average accuracy below 5%% '
                    'for %s consecutive rounds; results saved to %s',
                    method_name, round_index + 1, low_accuracy_rounds, summary_path,
                )
                break

        result = next(row for row in rows
                      if row.get('Method') == method_name and row.get('note', '') == note)
        logger.info('[%s] average across %s rounds: top1=%s%%',
                    method_name, round_index + 1, result['avg'])

        del adapter


def main():
    args = parse_args()
    cfg = build_runtime(args, 'benchmark_continual')
    run_benchmark_continual(args, cfg)


if __name__ == '__main__':
    main()
