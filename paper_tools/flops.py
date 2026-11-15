"""Count convolution/matrix-product FLOPs, including executed backward kernels.

One multiply-accumulate counts as two FLOPs. Normalization, elementwise
operations, reductions, indexing and data movement are excluded.
"""
from collections import Counter

from torch.utils._python_dispatch import TorchDispatchMode


class _FlopMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = func.overloadpacket.__name__
        flops = 0
        if op == 'convolution':
            x, weight = args[:2]
            if args[6]:
                flops = 2 * x.numel() * weight[0].numel()
            else:
                flops = 2 * out.numel() * weight[0].numel()
        elif op == 'convolution_backward':
            grad, x, weight = args[:3]
            forward = (2 * x.numel() * weight[0].numel() if args[7]
                       else 2 * grad.numel() * weight[0].numel())
            flops = forward * sum(args[10][:2])
        elif op in ('mm', 'bmm'):
            flops = 2 * out.numel() * args[0].shape[-1]
        elif op in ('addmm', 'baddbmm'):
            flops = 2 * out.numel() * args[1].shape[-1]
        elif op == 'addbmm':
            flops = 2 * args[1].shape[0] * out.numel() * args[1].shape[-1]
        if flops:
            self.counts[op] += int(flops)
        return out


class AdapterFlopCounter:
    """Scope counting to adapter calls, including adaptation inside forward()."""

    def __init__(self, module):
        self.mode = _FlopMode()
        self.calls = 0
        self.handles = [module.register_forward_pre_hook(self._pre),
                        module.register_forward_hook(self._post)]

    def _pre(self, *_):
        self.mode.__enter__()

    def _post(self, *_):
        self.mode.__exit__(None, None, None)
        self.calls += 1

    @property
    def total(self):
        return sum(self.mode.counts.values())

    def remove(self):
        for handle in self.handles:
            handle.remove()
