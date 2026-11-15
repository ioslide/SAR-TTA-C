"""Trim PKU-MMD Part I and map shared actions to the NTU60 classifier.

Run with the base Python (numpy, openpyxl and libarchive-c); evaluation uses the environment below.
The official archives and Actions.xlsx must be in ROOT/sources.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from openpyxl import load_workbook
import libarchive
import yaml


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('data/real_shifts/pku')
# PKU IDs 1..51 -> NTU IDs 1..60. PKU 29 concerns one's own pocket;
# NTU 57 concerns another person's pocket, so it is not a shared class.
NTU_IDS = (35, 4, 3, 33, 22, 10, 40, 1, 5, 2, 43, 56, 23, 58, 26, 55,
           27, 51, 24, 28, 53, 6, 29, 54, 31, 50, 52, 20, None, 11, 34,
           38, 8, 9, 21, 19, 15, 25, 32, 13, 7, 46, 45, 44, 47, 30, 49,
           14, 18, 37, 12)
SOURCE = 'http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def archive_files(path):
    with libarchive.file_reader(str(path)) as archive:
        for entry in archive:
            if entry.isfile:
                yield Path(entry.pathname).name, io.BytesIO(b''.join(entry.get_blocks()))


def prepare(root, config):
    root = root.resolve()
    sources = root / 'sources'
    target = root / 'pku_ntu60.mmap'
    if target.exists():
        raise FileExistsError(target)
    mapping = {i: ntu - 1 for i, ntu in enumerate(NTU_IDS, 1) if ntu is not None}
    rows = list(load_workbook(sources / 'Actions.xlsx', read_only=True).active.values)[1:]
    assert [int(row[0]) for row in rows] == list(range(1, 52))
    labels_by_video = {name: np.loadtxt(data, delimiter=',', dtype=int, ndmin=2)
                       for name, data in archive_files(sources / 'Label_PKUMMD.7z')}
    assert len(labels_by_video) == 1076
    metadata, excluded, offset, videos = [], [], 0, []
    with TemporaryDirectory(prefix='.pku-', dir=root) as temporary:
        temporary = Path(temporary)
        output = temporary / 'pku_ntu60.mmap'
        output.mkdir()
        with (output / 'keypoints.dat').open('wb') as handle:
            for name, content in archive_files(sources / 'Skeleton.7z'):
                videos.append(name)
                data = np.loadtxt(content, dtype=np.float32).reshape(-1, 2, 25, 3)
                labels = labels_by_video[name]
                for instance, (label, start, end, confidence) in enumerate(labels):
                    identifier = '{}_{:02d}'.format(Path(name).stem, instance)
                    if label not in mapping:
                        excluded.append(dict(frame_dir=identifier, reason='unshared_class', label=int(label)))
                        continue
                    # Follow MAMP's public PKU reader: data[start:end].
                    if not 0 <= start < end <= len(data):
                        excluded.append(dict(frame_dir=identifier, reason='invalid_interval', label=int(label),
                                             start=int(start), end=int(end), video_frames=len(data)))
                        continue
                    keypoint = data[start:end].transpose(1, 0, 2, 3).copy()
                    if not np.isfinite(keypoint).all() or not np.any(keypoint):
                        excluded.append(dict(frame_dir=identifier, reason='invalid_skeleton', label=int(label)))
                        continue
                    keypoint.tofile(handle)
                    metadata.append(dict(frame_dir=identifier, label=mapping[label],
                                         total_frames=int(end - start), source_video=Path(name).stem,
                                         source_label=int(label), source_start_frame=int(start),
                                         source_end_frame=int(end), source_confidence=int(confidence),
                                         mmap_index=len(metadata), mmap_offset=offset,
                                         mmap_size=keypoint.size, mmap_shape=list(keypoint.shape)))
                    offset += keypoint.size
                if len(videos) % 100 == 0:
                    print('{} / 1076 videos; {} clips'.format(len(videos), len(metadata)), flush=True)
        assert set(videos) == set(labels_by_video)
        metadata.sort(key=lambda item: item['frame_dir'])
        for i, item in enumerate(metadata):
            item['mmap_index'] = i
        assert set(item['label'] for item in metadata) == set(mapping.values())
        index = dict(format='protogcn_annotation_mmap', format_version=1, source=SOURCE,
                     dtype='float32', data_file='keypoints.dat', annotations=metadata,
                     split={'real_val': [item['frame_dir'] for item in metadata]})
        (output / 'index.json').write_text(json.dumps(index, separators=(',', ':')))
        output.rename(target)
    with (root / 'class_mapping.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pku_id_1based', 'pku_action', 'ntu60_label_0based'])
        writer.writerows((int(row[0]), row[1], mapping.get(int(row[0]), 'excluded')) for row in rows)
    manifest = dict(dataset='PKU-MMD Part I', source=SOURCE, raw_videos=len(videos),
                    samples=len(metadata), shared_classes=len(mapping), split='real_val',
                    selection='all Part I videos; shared classes; valid intervals; finite, nonempty skeletons',
                    stream_order='lexicographic video name, then original annotation row',
                    frame_intervals='zero-based, end exclusive; data[start:end], as in MAMP',
                    trimming_reference='https://github.com/maoyunyao/MAMP/blob/main/data/pku_v1/pku_gendata.py',
                    joints='Kinect v2 / NTU25, identity order',
                    coordinates='original camera-space XYZ in metres; no offline alignment',
                    dtype='float32', classifier='full 60-way NTU60 logits',
                    per_class_counts={str(label): sum(item['label'] == label for item in metadata)
                                      for label in sorted(mapping.values())},
                    excluded=excluded,
                    source_sha256={p.name: sha256(p) for p in sorted(sources.iterdir()) if p.is_file()})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    dataset = dict(name='ntu60_to_pku', split='real_val', data_root=str(root),
                   ann_file=str(target / 'index.json'), modality='j', num_classes=60,
                   shared_ntu60_labels=sorted(mapping.values()))
    config.write_text(yaml.safe_dump({'dataset': dataset}, sort_keys=False))
    print('{} clips, {} shared classes; {}'.format(len(metadata), len(mapping), config), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/dataset/ntu60_to_pku.yaml')
    args = parser.parse_args()
    prepare(args.root, args.config)
