#!/usr/bin/env python3
"""Export the evaluated checkpoint tensors without training metadata."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import torch

MODELS = {
    'ntu60_xsub/best_top1_acc_epoch_149.pth': 'https://huggingface.co/firework8/ProtoGCN',
    'ntu60_xview/best_top1_acc_epoch_149.pth': 'https://huggingface.co/firework8/ProtoGCN',
    'ntu120_xsub/best_top1_acc_epoch_141.pth': 'https://huggingface.co/firework8/ProtoGCN/resolve/main/ntu120_xsub/j_1/best_top1_acc_epoch_141.pth',
    'stgcnpp_xview/j.pth': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/j.pth',
    'ctrgcn_xview/j.pth': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/j.pth',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('release_assets/checkpoints.tar.gz'))
    parser.add_argument('--manifest', type=Path, default=Path('checkpoints_manifest.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.open('wb') as handle:
        with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for name, source in MODELS.items():
                    original = (args.source / name).read_bytes()
                    checkpoint = torch.load(io.BytesIO(original), map_location='cpu')
                    state = checkpoint.get('state_dict', checkpoint)
                    assert all(torch.is_tensor(tensor) for tensor in state.values())
                    payload = io.BytesIO()
                    torch.save({'state_dict': state}, payload)
                    raw = payload.getvalue()
                    restored = torch.load(io.BytesIO(raw), map_location='cpu')['state_dict']
                    assert all(torch.equal(state[k], restored[k]) for k in state)
                    info = tarfile.TarInfo('checkpoints/' + name)
                    info.mode, info.size = 0o644, len(raw)
                    archive.addfile(info, io.BytesIO(raw))
                    records.append(dict(file='checkpoints/' + name, source=source,
                                        original_sha256=hashlib.sha256(original).hexdigest(),
                                        release_sha256=hashlib.sha256(raw).hexdigest(),
                                        bytes=len(raw), tensors=len(state)))
    args.manifest.write_text(json.dumps({'format': 'state_dict_only', 'models': records}, indent=2) + '\n')
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + '.SHA256SUMS').write_text(f'{digest}  {args.output.name}\n')
    print(f'Exported and verified {len(records)} tensor-identical checkpoints.')


if __name__ == '__main__':
    main()
