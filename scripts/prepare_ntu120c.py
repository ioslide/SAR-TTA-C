"""Create NTU60/120-C test-only mmap annotations with the existing v2 corruptions."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import load_yaml
from core.runtime import ALL_CORRUPTIONS
from scripts.convert_annotations_to_mmap import _json_value
from scripts.generate_offline_corruptions import V2_CORRUPTIONS, V2_FORMAT_VERSION, _rng_for_item

NTU120_CORRUPTIONS = ALL_CORRUPTIONS[:-1]


def write_domain(target, annotations, split, source, name, severity, seed):
    index_path = target / 'index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        assert len(index['annotations']) == len(annotations) and split in index['split']
        return dict(name=name, index=str(index_path), samples=len(annotations), status='existing')
    total = sum(item['keypoint'].size for item in annotations)
    with TemporaryDirectory(prefix='.' + target.name + '-', dir=target.parent) as temporary:
        directory = Path(temporary)
        data = np.memmap(directory / 'keypoints.dat', mode='w+', dtype=np.float16, shape=(total,))
        metadata, offset = [], 0
        for i, item in enumerate(annotations):
            original = item['keypoint']
            keypoint = original if name == 'clean' else V2_CORRUPTIONS[name][2](
                original.astype(np.float32), _rng_for_item(seed, name, item['frame_dir']), severity)
            assert keypoint.shape == original.shape and np.isfinite(keypoint).all(), item['frame_dir']
            data[offset:offset + keypoint.size] = keypoint.reshape(-1)
            record = {k: _json_value(v) for k, v in item.items() if k != 'keypoint'}
            record.update(mmap_index=i, mmap_offset=offset, mmap_size=keypoint.size,
                          mmap_shape=list(keypoint.shape))
            metadata.append(record); offset += keypoint.size
        data.flush(); del data
        index = dict(format='protogcn_annotation_mmap', format_version=1, source=str(source),
                     dtype='float16', data_file='keypoints.dat', annotations=metadata,
                     split={split: [item['frame_dir'] for item in annotations]},
                     offline_corruption=dict(format_version=V2_FORMAT_VERSION, name=name,
                                             severity=severity, seed=seed, source=str(source)))
        (directory / 'index.json').write_text(json.dumps(index, separators=(',', ':')))
        directory.rename(target)
    return dict(name=name, index=str(index_path), samples=len(annotations), status='generated',
                bytes=total * np.dtype(np.float16).itemsize)


def prepare(dataset_config):
    cfg = load_yaml(dataset_config)['dataset']
    source = Path(cfg['ann_file']); output = Path(cfg['offline_corruption_dir'])
    split = cfg['split']; severity = cfg['offline_corruption_severity']; seed = cfg['offline_corruption_seed']
    num_classes = cfg.get('num_classes', 60)
    corruptions = ALL_CORRUPTIONS if num_classes == 60 else NTU120_CORRUPTIONS
    name = 'NTU60-C' if num_classes == 60 else 'NTU120-C'
    output.mkdir(parents=True, exist_ok=True)
    print('Loading ' + str(source), flush=True)
    with source.open('rb') as handle:
        payload = pickle.load(handle)
    selected = set(payload['split'][split])
    annotations = [a for a in payload['annotations'] if a['frame_dir'] in selected]
    del payload
    assert len(annotations) == len(selected)
    assert set(int(a['label']) for a in annotations) == set(range(num_classes))
    manifest = dict(dataset=name, source=str(source), split=split,
                    num_classes=num_classes, samples=len(annotations),
                    severity=severity, seed=seed, corruption_order=corruptions,
                    transform_sha256=hashlib.sha256(Path(__file__).with_name(
                        'generate_offline_corruptions.py').read_bytes()).hexdigest(), domains=[])
    manifest_path = output / 'manifest.json'
    for name in (['clean'] if num_classes == 120 else []) + corruptions:
        filename = (f'{source.stem}_clean.pkl.mmap' if name == 'clean' else
                    f'{source.stem}_corrupt_{name}_s{severity}_seed{seed}.pkl.mmap')
        record = write_domain(output / filename, annotations, split, source, name, severity, seed)
        manifest['domains'].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(record), flush=True)
    print('NTU-C ready: ' + str(manifest_path), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', default='configs/dataset/ntu120_xsub.yaml')
    prepare(parser.parse_args().dataset_config)
