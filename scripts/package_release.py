#!/usr/bin/env python3
"""Package a prepared NTU-C directory with portable metadata and split archives."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


class SplitWriter:
    def __init__(self, directory, name, part_bytes):
        self.directory, self.name, self.limit = directory, name, part_bytes
        self.handle = None
        self.parts = []
        self.size = 0

    def write(self, data):
        original_size = len(data)
        while data:
            if self.handle is None or self.size == self.limit:
                self.close_part()
                self.path = self.directory / f'{self.name}.part{len(self.parts):03d}'
                self.handle = self.path.open('wb')
                self.digest = hashlib.sha256()
                self.size = 0
            chunk = data[:self.limit - self.size]
            self.handle.write(chunk)
            self.digest.update(chunk)
            self.size += len(chunk)
            data = data[len(chunk):]
        return original_size

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close_part(self):
        if self.handle:
            self.handle.close()
            self.parts.append((self.path.name, self.digest.hexdigest()))
            self.handle = None


class HashedReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def portable(value, root):
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, root) for v in value]
    if isinstance(value, str) and value.startswith('/'):
        path = Path(value)
        return str(path.relative_to(root)) if root in path.parents else path.name
    return value


def add_member(archive, name, size, handle):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    archive.addfile(info, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('release_assets'))
    parser.add_argument('--catalog-dir', type=Path, default=Path('data/manifests'))
    parser.add_argument('--part-mib', type=int, default=1024)
    args = parser.parse_args()
    root = args.source.resolve()
    manifest = portable(json.loads((root / 'manifest.json').read_text()), root)
    # NTU120-C uses the 17 individual corruptions, plus the optional clean reference.
    if manifest['num_classes'] == 120:
        manifest['corruption_order'] = [x for x in manifest['corruption_order'] if x != 'mixed_failure']
        manifest['domains'] = [x for x in manifest['domains'] if x['name'] != 'mixed_failure']
    args.output.mkdir(parents=True, exist_ok=True)
    args.catalog_dir.mkdir(parents=True, exist_ok=True)
    name = root.name + '.tar.gz'
    if list(args.output.glob(name + '.part*')):
        raise FileExistsError('Remove the existing parts before packaging again.')
    writer = SplitWriter(args.output, name, args.part_mib * 1024**2)
    prefix = 'data/nturgbd/' + root.name
    files = []
    with gzip.GzipFile(filename='', fileobj=writer, mode='wb', compresslevel=1, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for domain in manifest['domains']:
                relative = Path(domain['index']).parent
                index_path = root / relative / 'index.json'
                index = portable(json.loads(index_path.read_text()), root)
                assert len(index['annotations']) == manifest['samples']
                data_path = root / relative / index['data_file']
                itemsize = {'float16': 2, 'float32': 4}[index['dtype']]
                offset = 0
                for item in index['annotations']:
                    assert item['mmap_offset'] == offset
                    product = 1
                    for dim in item['mmap_shape']:
                        product *= dim
                    assert product == item['mmap_size']
                    offset += product
                assert offset * itemsize == data_path.stat().st_size
                for filename, payload in [('index.json', json.dumps(index, separators=(',', ':')).encode()),
                                          (index['data_file'], None)]:
                    rel = str(relative / filename)
                    if payload is not None:
                        add_member(archive, prefix + '/' + rel, len(payload), io.BytesIO(payload))
                        digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
                    else:
                        with data_path.open('rb') as handle:
                            hashed = HashedReader(handle)
                            size = data_path.stat().st_size
                            add_member(archive, prefix + '/' + rel, size, hashed)
                            digest = hashed.digest.hexdigest()
                    files.append(dict(path=rel, bytes=size, sha256=digest))
                print(f'{root.name}: {domain["name"]} ({manifest["samples"]} samples)', flush=True)
            manifest['files'] = files
            manifest['release_generator_sha256'] = hashlib.sha256(
                Path(__file__).with_name('generate_offline_corruptions.py').read_bytes()).hexdigest()
            payload = (json.dumps(manifest, indent=2) + '\n').encode()
            add_member(archive, prefix + '/manifest.json', len(payload), io.BytesIO(payload))
    writer.close_part()
    (args.catalog_dir / (root.name + '.json')).write_bytes(payload)
    checksum_file = args.output / (name + '.SHA256SUMS')
    checksum_file.write_text(''.join(f'{digest}  {part}\n' for part, digest in writer.parts))
    print(f'Complete: {len(writer.parts)} parts; checksums: {checksum_file.name}', flush=True)


if __name__ == '__main__':
    main()
