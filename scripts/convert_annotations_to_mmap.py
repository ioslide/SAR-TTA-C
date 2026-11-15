#!/usr/bin/env python
"""Convert ProtoGCN annotation PKLs into lazy NumPy memmap datasets.

The generated directory sits next to the source file:
``<annotation>.mmap/index.json`` and ``keypoints.dat``.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def convert(source, output=None, overwrite=False):
    source = Path(source)
    output = Path(output) if output else Path(f'{source}.mmap')
    index_path = output / 'index.json'
    data_path = output / 'keypoints.dat'
    if index_path.exists() or data_path.exists():
        if not overwrite:
            print(f'skip existing: {index_path}')
            return
        for path in (index_path, data_path):
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    print(f'loading {source}', flush=True)
    with source.open('rb') as handle:
        payload = pickle.load(handle)
    annotations = payload.get('annotations')
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f'{source} has no annotations list')

    total_values = sum(int(np.asarray(item['keypoint']).size) for item in annotations)
    mmap = np.memmap(data_path, mode='w+', dtype=np.float16, shape=(total_values,))
    metadata = []
    offset = 0
    for index, item in enumerate(annotations):
        keypoint = np.asarray(item['keypoint'], dtype=np.float16)
        size = int(keypoint.size)
        mmap[offset:offset + size] = keypoint.reshape(-1)
        record = {
            key: _json_value(value)
            for key, value in item.items()
            if key != 'keypoint'
        }
        record.update({
            'mmap_index': index,
            'mmap_offset': offset,
            'mmap_size': size,
            'mmap_shape': list(keypoint.shape),
        })
        metadata.append(record)
        offset += size
        if (index + 1) % 10000 == 0:
            print(f'indexed {index + 1}/{len(annotations)}', flush=True)
    mmap.flush()
    del mmap

    index = {
        'format': 'protogcn_annotation_mmap',
        'format_version': FORMAT_VERSION,
        'source': str(source),
        'dtype': 'float16',
        'data_file': data_path.name,
        'annotations': metadata,
        'split': _json_value(payload.get('split', {})),
        'offline_corruption': _json_value(payload.get('offline_corruption', {})),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=True, separators=(',', ':')) + '\n')
    print(f'written {index_path} ({len(metadata)} annotations, {data_path.stat().st_size} bytes)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for source in args.sources:
        convert(source, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
