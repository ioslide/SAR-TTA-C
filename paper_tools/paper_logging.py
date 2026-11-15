"""Stream logging, adapter wrappers and timing used by the [paper] hooks in run.py / L-CS_run.py.

Nothing here changes what the adapters compute. For every evaluated domain the runner calls

    stream_log.begin(adapter, round_index, domain_index, tag)
    with GpuTimer() as timer:
        outputs = evaluate_on_loader(adapter, data_loader, test_cfg)
    ...
    stream_log.end(adapter, outputs, dataset, timer, top1=top1_acc * 100.0)

and ``end`` writes one compressed .npz file

    $PAPER_LOG_DIR/<kind>/<method>/<note>/r<round>_d<domain>_<tag>.npz
        pred  [n] int16      predicted class          label [n] int16   ground truth (if found)
        conf  [n] float16    max probability          bmean [nb, C]     mean probability per batch
        bsize [nb] int16     batch sizes              tr_*  [nb]        per-batch trace of the adapter
plus one line in ``<run dir>/index.jsonl`` (accuracy, timing, memory). paper_tools/collect_runs.py
turns these files into the data of every table and figure of the paper.

Environment
  PAPER_LOG_DIR     root of the logs (empty/unset: logging disabled, hooks become no-ops)
  PAPER_TIMING=1    time every adapter forward with CUDA synchronisation (Tab. S4)
  PAPER_FLOPS=1     count convolution/matmul FLOPs, including backward (2 per MAC)
  PLUGIN_ATC=1      wrap the adapter with Arrow-of-Time Canonicalization (Tab. 4, bottom)
  PAPER_PROBE_GAP=1 wrap the adapter to also evaluate tau(x) without adaptation (Fig. 2c)
  ATC_RHO           vote momentum of the plug-in (default 0.9)
  ASTRA_TIME_DIM    dimension reversed by tau (default 2)
  SKIP_PKL=1        do not write the large .pkl result files (the .npz logs keep predictions)
  PAPER_CSV_TAG     if set, results go to <csv stem>__<tag>.csv (safe parallel runs)
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TIME_DIM = int(os.environ.get('ASTRA_TIME_DIM', '2'))
EPS = 1e-12


# ====================================================================== small helpers
def skip_pkl(round_index=0):
    """Keep the original .pkl dumps unless SKIP_PKL=1 (round 0 is always kept if SKIP_PKL=rest)."""
    mode = os.environ.get('SKIP_PKL', '0')
    if mode == 'rest':
        return round_index > 0
    return mode == '1'


def pkl_name(method, note, tag, severity, round_index=None):
    note_part = f'_{note}' if note else ''
    round_part = f'_round_{round_index}' if round_index is not None else ''
    return f'{method}{note_part}{round_part}_{tag}_s{severity}.pkl'


def paper_csv_path(path):
    tag = os.environ.get('PAPER_CSV_TAG', '')
    if not tag:
        return path
    path = Path(path)
    return path.with_name(f'{path.stem}__{tag}{path.suffix}')


def dataset_labels(dataset):
    """Ground-truth labels in loader order (test loaders are not shuffled)."""
    for attr in ('video_infos', 'data_infos', 'annotations'):
        infos = getattr(dataset, attr, None)
        if infos:
            try:
                return np.asarray([int(info['label']) for info in infos], dtype=np.int64)
            except (KeyError, TypeError, ValueError):
                pass
    try:
        return np.asarray([int(dataset[i]['label']) for i in range(len(dataset))], dtype=np.int64)
    except Exception:  # noqa: BLE001
        return None


def _to_numpy_probs(outputs):
    arr = np.asarray([np.asarray(o, dtype=np.float32).reshape(-1) for o in outputs], dtype=np.float32)
    rows = arr.sum(axis=1)
    if arr.min() < -1e-6 or np.abs(rows - 1.0).max() > 1e-3:      # logits -> probabilities
        arr = arr - arr.max(axis=1, keepdims=True)
        arr = np.exp(arr)
        arr /= arr.sum(axis=1, keepdims=True)
    return arr


def _entropy_from_output(out):
    out = out.detach().float()
    if out.dim() > 2:
        out = out.reshape(out.shape[0], -1)
    rows = out.sum(dim=1)
    if out.min() >= 0 and torch.allclose(rows, torch.ones_like(rows), atol=1e-3):
        p = out.clamp_min(EPS)
        return -(p * p.log()).sum(dim=1)
    return -(out.softmax(dim=1) * out.log_softmax(dim=1)).sum(dim=1)


def _finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


class GpuTimer:
    """Wall-clock time and peak GPU memory of a block (CUDA-synchronised)."""

    def __enter__(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
            self.peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        else:
            self.peak_mem_gb = float('nan')
        self.seconds = time.perf_counter() - self._t0
        return False


# ====================================================================== adapter wrappers
class _Wrapper(nn.Module):
    """Delegates every unknown attribute to the wrapped adapter."""

    def __init__(self, inner):
        super().__init__()
        if isinstance(inner, nn.Module):
            self.inner = inner
        else:
            object.__setattr__(self, '_inner_obj', inner)
        self._trace = {}
        self._stream_tag = None

    def _inner(self):
        inner = self.__dict__.get('_inner_obj')
        if inner is None:
            inner = self.__dict__.get('_modules', {}).get('inner')
        return inner

    @property
    def wrapped(self):
        return self._inner()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._inner()
            if inner is None:
                raise
            return getattr(inner, name)

    def _model(self):
        inner = self.wrapped
        model = getattr(inner, 'model', None)
        return model if callable(model) else inner

    def set_stream_position(self, round_index, domain_index, tag):
        self._stream_tag = tag
        fn = getattr(self.wrapped, 'set_stream_position', None)
        if callable(fn):
            fn(round_index, domain_index, tag)

    def _tr(self, name, value):
        if not torch.is_tensor(value):
            value = torch.tensor(float(value))
        self._trace.setdefault(name, []).append(value.detach().float().reshape(()))

    def pop_paper_trace(self):
        out = {k: torch.stack(v).cpu().numpy() for k, v in self._trace.items()}
        self._trace = {}
        return out


class ATCPlugin(_Wrapper):
    """Arrow-of-Time Canonicalization in front of any adapter (Tab. 4, bottom).

    Same rule as ASTRA: both orientations are scored with the current parameters, the sign of
    the batch-entropy gap enters an EMA with momentum rho, and the adapter only ever sees
    (and adapts on) the canonical view.
    """

    def __init__(self, inner, rho=0.9):
        super().__init__(inner)
        self.rho = float(rho)
        self.rho_c = round(1.0 - self.rho, 12)
        self.score = 0.0

    def forward(self, keypoint, *args, **kwargs):
        reversed_keypoint = keypoint.flip([TIME_DIM])
        reverse = self.score > 0.0
        canonical, alternate = (reversed_keypoint, keypoint) if reverse else (keypoint, reversed_keypoint)
        with torch.no_grad():
            h_alt = _entropy_from_output(self._model()(alternate)).mean()
        outputs = self.wrapped(canonical, *args, **kwargs)
        h_can = _entropy_from_output(outputs).mean()
        preference = (h_can - h_alt).sign().item()
        self.score = self.rho * self.score + self.rho_c * (-preference if reverse else preference)
        self._tr('o', float(reverse))
        self._tr('s', self.score)
        self._tr('h_can', h_can)
        self._tr('h_alt', h_alt)
        return outputs


class GapProbe(_Wrapper):
    """Fig. 2(c): mean entropy of x and of tau(x) under a non-adapting adapter (e.g. Source)."""

    def forward(self, keypoint, *args, **kwargs):
        outputs = self.wrapped(keypoint, *args, **kwargs)
        with torch.no_grad():
            outputs_rev = self.wrapped(keypoint.flip([TIME_DIM]), *args, **kwargs)
        self._tr('h_x', _entropy_from_output(outputs).mean())
        self._tr('h_r', _entropy_from_output(outputs_rev).mean())
        return outputs


def maybe_wrap_adapter(adapter, logger=None):
    if os.environ.get('PAPER_PROBE_GAP', '0') == '1':
        if logger:
            logger.info('[paper] Fig.2(c) probe: evaluating x and tau(x) (no adaptation expected)')
        adapter = GapProbe(adapter)
    if os.environ.get('PLUGIN_ATC', '0') == '1':
        rho = float(os.environ.get('ATC_RHO', '0.9'))
        if logger:
            logger.info('[paper] wrapping adapter with Arrow-of-Time Canonicalization (rho=%s)', rho)
        adapter = ATCPlugin(adapter, rho=rho)
    return adapter


maybe_wrap_atc = maybe_wrap_adapter  # backwards-compatible name


# ====================================================================== stream logger
class _ForwardTimer:
    def __init__(self, module):
        self.total, self.count, self._t0 = 0.0, 0, None
        self.handles = [module.register_forward_pre_hook(self._pre),
                        module.register_forward_hook(self._post)]

    def _pre(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def _post(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._t0 is not None:
            self.total += time.perf_counter() - self._t0
            self.count += 1
            self._t0 = None

    def remove(self):
        for h in self.handles:
            h.remove()


class StreamLogger:
    def __init__(self, root, kind, method, note, batch_size):
        self.root = Path(root) if root else None
        self.kind, self.method, self.note = kind, method, note or 'default'
        self.batch_size = int(batch_size)
        self.timing = os.environ.get('PAPER_TIMING', '0') == '1'
        self.flops = os.environ.get('PAPER_FLOPS', '0') == '1'
        self._pos = None
        self._fwd = None
        self._flops = None
        if self.root is not None:
            self.dir = self.root / kind / method / self.note
            self.dir.mkdir(parents=True, exist_ok=True)
            # Restart adapter state from scratch while preserving prior measurements.
            if any(self.dir.glob('*.npz')) or (self.dir / 'index.jsonl').exists():
                archive = self.root / 'archive' / kind / method
                archive.mkdir(parents=True, exist_ok=True)
                self.dir.rename(archive / f'{self.note}_{time.time_ns()}')
                self.dir.mkdir()

    @classmethod
    def from_env(cls, kind, method, note, batch_size):
        return cls(os.environ.get('PAPER_LOG_DIR', ''), kind, method, note, batch_size)

    @property
    def enabled(self):
        return self.root is not None

    def begin(self, adapter, round_index, domain_index, tag):
        self._pos = (int(round_index), int(domain_index), str(tag))
        fn = getattr(adapter, 'set_stream_position', None)
        if callable(fn):
            fn(round_index, domain_index, tag)
        if self.enabled and self.timing and isinstance(adapter, nn.Module):
            self._fwd = _ForwardTimer(adapter)
        if self.enabled and self.flops:
            from paper_tools.flops import AdapterFlopCounter
            self._flops = AdapterFlopCounter(adapter)

    def end(self, adapter, outputs, dataset, timer=None, top1=None):
        fwd = self._fwd
        if fwd is not None:
            fwd.remove()
            self._fwd = None
        flops = self._flops
        if flops is not None:
            flops.remove()
            self._flops = None
        trace = {}
        fn = getattr(adapter, 'pop_paper_trace', None)
        if callable(fn):
            trace = fn() or {}
        if not self.enabled:
            return
        round_index, domain_index, tag = self._pos
        probs = _to_numpy_probs(outputs)
        n, num_classes = probs.shape
        pred = probs.argmax(axis=1)
        labels = dataset_labels(dataset)
        if labels is not None and len(labels) != n:
            labels = None
        nb = int(math.ceil(n / self.batch_size))
        bsize = np.array([min(self.batch_size, n - b * self.batch_size) for b in range(nb)], dtype=np.int16)
        bmean = np.stack([probs[b * self.batch_size:(b + 1) * self.batch_size].mean(axis=0)
                          for b in range(nb)]).astype(np.float32)
        arrays = dict(pred=pred.astype(np.int16), conf=probs.max(axis=1).astype(np.float16),
                      bmean=bmean, bsize=bsize)
        if labels is not None:
            arrays['label'] = labels.astype(np.int16)
        for key, value in trace.items():
            arrays[f'tr_{key}'] = np.asarray(value, dtype=np.float32)
        fname = f'r{round_index:02d}_d{domain_index:02d}_{tag}.npz'
        np.savez_compressed(self.dir / fname, **arrays)
        acc = float((pred == labels).mean() * 100.0) if labels is not None else None
        record = dict(kind=self.kind, method=self.method, note=self.note, round=round_index,
                      domain=domain_index, tag=tag, n=int(n), num_classes=int(num_classes),
                      batch_size=self.batch_size, n_batches=nb, file=fname,
                      top1=None if top1 is None else float(top1), top1_from_labels=acc,
                      seconds=getattr(timer, 'seconds', None),
                      peak_mem_gb=_finite(getattr(timer, 'peak_mem_gb', None)),
                      fwd_seconds=None if fwd is None else fwd.total,
                      fwd_calls=None if fwd is None else fwd.count,
                      time=time.strftime('%Y-%m-%d %H:%M:%S'))
        if flops is not None:
            record.update(flops=flops.total, flop_calls=flops.calls,
                          gflops_per_batch=flops.total / flops.calls / 1e9,
                          flops_by_op=dict(flops.mode.counts),
                          flop_convention='conv_matmul_forward_backward_2_per_mac')
        with open(self.dir / 'index.jsonl', 'a') as fh:
            fh.write(json.dumps(record) + '\n')
