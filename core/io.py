import csv
import fcntl
import json
import logging
from datetime import datetime
from pathlib import Path


SUMMARY_FIELDS = [
    'folder', 'dataset', 'split', 'modality', 'protocol', 'continual_adaptation',
    'random_order_seed', 'ann_file', 'checkpoint', 'batch_size', 'num_clips',
    'method', 'corruption', 'severity', 'top1_acc', 'top5_acc', 'tent_lr',
    'tent_steps', 'bn_mode', 'note', 'notes', 'source_model', 'source_config',
    'run_timestamp', 'git_commit', 'data_root', 'status', 'conda_env'
]

WIDE_BASE_FIELDS = [
    'Method', 'Acc', 'Severity', 'Seed', 'BatchSize', 'NOTE'
]


def _accuracy_fraction(value):
    """Normalize a stored accuracy to a fraction for arithmetic."""
    if value in ('', None):
        return None
    number = float(value)
    return number / 100.0 if abs(number) > 1.0 else number


def _accuracy_percent(value):
    fraction = _accuracy_fraction(value)
    return '' if fraction is None else f'{fraction * 100.0:.2f}'


def timestamp():
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def setup_logger(log_path):
    ensure_dir(Path(log_path).parent)
    logger = logging.getLogger('protogcn-runner')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def write_summary_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        formatted = []
        for row in rows:
            item = dict(row)
            for field in ('top1_acc', 'top5_acc'):
                if item.get(field, '') not in ('', None):
                    item[field] = _accuracy_percent(item[field])
            formatted.append(item)
        writer.writerows(formatted)


def write_corruption_wide_csv(path, rows, corruption_order=None, note_override=None):
    path = Path(path)
    ensure_dir(path.parent)
    corruptions = _ordered_corruptions(rows, corruption_order)
    fields = WIDE_BASE_FIELDS[:]
    fields.extend(corruptions)
    fields.append('Avg')

    grouped = {}
    for row in rows:
        corruption = row.get('corruption', '')
        if not corruption:
            continue
        for acc_name, value_field in [('top1', 'top1_acc'), ('top5', 'top5_acc')]:
            key = (
                row.get('dataset', ''),
                row.get('split', ''),
                row.get('modality', ''),
                row.get('protocol', ''),
                row.get('random_order_seed', ''),
                row.get('checkpoint', ''),
                row.get('batch_size', ''),
                row.get('method', ''),
                row.get('severity', ''),
                acc_name,
            )
            if key not in grouped:
                grouped[key] = {
                    'Method': row.get('method', ''),
                    'Acc': acc_name,
                    'Severity': row.get('severity', ''),
                    'Seed': row.get('random_order_seed', ''),
                    'BatchSize': row.get('batch_size', ''),
                    'NOTE': note_override if note_override is not None else row.get('note', ''),
                }
            grouped[key][corruption] = _accuracy_percent(row.get(value_field, ''))

    for wide_row in grouped.values():
        values = []
        for corruption in corruptions:
            value = wide_row.get(corruption, '')
            if value == '':
                continue
            normalized = _accuracy_fraction(value)
            if normalized is not None:
                values.append(normalized)
        if values:
            wide_row['Avg'] = f'{sum(values) / len(values) * 100.0:.2f}'

    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(grouped.values())


def _ordered_corruptions(rows, corruption_order=None):
    ordered = []
    for name in corruption_order or []:
        if name and name not in ordered:
            ordered.append(name)
    for row in rows:
        name = row.get('corruption', '')
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def read_summary_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


INDEPENDENT_TOP1_FIELDS = [
    'Method', 'dataset_name', 'seed', 'severity', 'batch_size', 'corruption',
    'top1_acc',
]


def read_independent_top1_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_independent_top1_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    formatted_rows = []
    for row in rows:
        item = dict(row)
        if item.get('top1_acc', '') not in ('', None):
            item['top1_acc'] = _accuracy_percent(item['top1_acc'])
        formatted_rows.append(item)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=INDEPENDENT_TOP1_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(formatted_rows)


def independent_top1_completed(rows):
    return {
        (row.get('Method', ''), row.get('corruption', ''))
        for row in rows
        if row.get('Method') and row.get('corruption')
    }


def update_independent_top1_csv(path, method, corruption, accuracy,
                                seed=None, dataset_name=None, severity=None,
                                batch_size=None):
    """Append or replace one independent method/corruption result row."""
    path = Path(path)
    ensure_dir(path.parent)
    lock_path = path.with_name(f'{path.name}.lock')
    with lock_path.open('a+') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = read_independent_top1_csv(path)
        if dataset_name is not None:
            for item in rows:
                item.setdefault('dataset_name', dataset_name)
        if seed is not None:
            for item in rows:
                item.setdefault('seed', seed)
        replacement = {
            'Method': method,
            'dataset_name': '' if dataset_name is None else dataset_name,
            'seed': '' if seed is None else seed,
            'severity': '' if severity is None else severity,
            'batch_size': '' if batch_size is None else batch_size,
            'corruption': corruption,
            'top1_acc': accuracy,
        }
        replaced = False
        for index, row in enumerate(rows):
            if row.get('Method') == method and row.get('corruption') == corruption:
                rows[index] = replacement
                replaced = True
                break
        if not replaced:
            rows.append(replacement)
        write_independent_top1_csv(path, rows)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return rows


def read_top1_matrix_csv(path):
    """Read the method-by-corruption Top-1 matrix used by the benchmark."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_top1_matrix_csv(path, rows, corruption_order=None):
    """Write a Top-1 matrix whose cells and averages are percentages."""
    path = Path(path)
    ensure_dir(path.parent)
    metadata = ['Method', 'dataset_name', 'seed', 'severity', 'batch_size', 'note']
    corruptions = []
    for row in rows:
        for name in row:
            if name not in metadata and name != 'avg' and name not in corruptions:
                corruptions.append(name)
    for name in corruption_order or []:
        if name and name not in metadata and name != 'avg' and name not in corruptions:
            corruptions.append(name)

    fields = metadata + corruptions
    if any('avg' in row for row in rows):
        fields.append('avg')
    formatted_rows = []
    for row in rows:
        item = dict(row)
        for name in corruptions:
            if item.get(name, '') not in ('', None):
                item[name] = f'{float(item[name]):.2f}'
        if item.get('avg', '') not in ('', None):
            item['avg'] = f'{float(item["avg"]):.2f}'
        formatted_rows.append(item)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(formatted_rows)


def update_top1_matrix_csv(path, method, corruption, accuracy,
                           corruption_order=None, severity=None,
                           batch_size=None, seed=None, dataset_name=None,
                           note=''):
    """Merge a fractional accuracy into the percentage-valued benchmark matrix."""
    path = Path(path)
    ensure_dir(path.parent)
    lock_path = path.with_name(f'{path.name}.lock')
    with lock_path.open('a+') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = read_top1_matrix_csv(path)
        if dataset_name is not None:
            for item in rows:
                item.setdefault('dataset_name', dataset_name)
        if seed is not None:
            for item in rows:
                item.setdefault('seed', seed)
        row = next((item for item in rows
                    if item.get('Method') == method
                    and item.get('note', '') == note), None)
        if row is None:
            row = {'Method': method, 'note': note}
            rows.append(row)
        row['note'] = note
        if dataset_name is not None:
            row['dataset_name'] = dataset_name
        if seed is not None:
            row['seed'] = seed
        if severity is not None:
            row['severity'] = severity
        if batch_size is not None:
            row['batch_size'] = batch_size
        row[corruption] = float(accuracy) * 100.0
        values = []
        for name in corruption_order or []:
            value = row.get(name, '')
            if value not in ('', None):
                values.append(float(value))
        if values:
            row['avg'] = sum(values) / len(values)
        write_top1_matrix_csv(path, rows, corruption_order)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return rows


def top1_matrix_completed(rows):
    """Return populated method/corruption cells for resumable execution."""
    completed = set()
    for row in rows:
        method = row.get('Method', '')
        if not method:
            continue
        for corruption, value in row.items():
            if corruption not in {
                'Method', 'dataset_name', 'seed', 'severity', 'batch_size',
                'note', 'avg',
            } and value not in ('', None):
                completed.add((method, row.get('note', ''), corruption))
    return completed


def dump_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
