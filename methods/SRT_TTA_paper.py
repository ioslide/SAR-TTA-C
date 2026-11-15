"""SRT: paper method, component ablations and instrumentation."""
import math
import os
from copy import deepcopy

import torch
from torch import nn

from paper_tools.variants import parse_note

EPS = 1e-12
TIME_DIM = int(os.environ.get('ASTRA_TIME_DIM', '2'))
REVERSE_TAG = os.environ.get('ASTRA_REVERSE_TAG', 'temporal_reverse')


def shannon_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def min_max_scale(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min()
    span = values.max() - minimum
    return torch.where(
        span > EPS,
        (values - minimum) / span.clamp_min(EPS),
        torch.zeros_like(values),
    )


def _complement(rho: float) -> float:
    # 1 - 0.9 == 0.09999999999999998 in float64; round so that rho = 0.9 gives exactly 0.1
    return round(1.0 - rho, 12)


class SRT_TTA_paper(nn.Module):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, num_classes: int,
                 variant: dict = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.register_buffer(
            "class_probs_ema",
            torch.full(
                (num_classes,), 1.0 / num_classes,
                device=next(model.parameters()).device,
            ),
        )
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self._reverse_score = 0.0

        v = dict(parse_note('full'))
        v.update(variant or {})
        self.variant = v
        self.alpha = float(v['alpha'])
        self.beta = float(v['beta'])
        self.rho_pi = float(v['rho_pi'])
        self.rho_pi_c = _complement(self.rho_pi)
        self.rho_s = float(v['rho_s'])
        self.rho_s_c = _complement(self.rho_s)
        self.num_classes = num_classes
        self._log_c = math.log(num_classes)
        self._stream_tag = None
        self._batch_counter = 0
        self._trace_on = os.environ.get('ASTRA_TRACE', '1') != '0'
        self._trace = {}

    # ------------------------------------------------------------------ instrumentation
    def set_stream_position(self, round_index, domain_index, tag):
        self._stream_tag = tag

    def _tr(self, name, value):
        if self._trace_on:
            if not torch.is_tensor(value):
                value = torch.tensor(float(value))
            self._trace.setdefault(name, []).append(value.detach().float().reshape(()))

    def pop_paper_trace(self):
        out = {k: torch.stack(vals).cpu().numpy() for k, vals in self._trace.items()}
        self._trace = {}
        return out

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.class_probs_ema.fill_(1.0 / self.class_probs_ema.numel())
        self._reverse_score = 0.0

    # ------------------------------------------------------------------ SREM
    @torch.no_grad()
    def _sample_weights(self, logits: torch.Tensor):
        probs = logits.softmax(dim=1)
        p_embed = probs.pow(self.alpha / 2)
        prior_embed = self.class_probs_ema.pow(self.alpha / 2)
        affinity = (p_embed * prior_embed).sum(dim=1) / (
            p_embed.norm(dim=1).clamp_min(EPS)
            * prior_embed.norm().clamp_min(EPS)
        )

        anti_sink = min_max_scale(1.0 - affinity)
        entropy = shannon_entropy(logits)
        confidence = min_max_scale(-entropy)
        if self.variant['gate']:
            selected = anti_sink >= anti_sink.mean()
        else:
            selected = torch.ones_like(anti_sink, dtype=torch.bool)
        if self.variant['weight'] == 'coupled':
            weights = (self.beta * anti_sink * confidence).exp()
        elif self.variant['weight'] == 'conf':
            weights = (self.beta * confidence).exp()
        else:
            weights = torch.ones_like(anti_sink)
        weights = weights.masked_fill(~selected, 0.0)

        self.class_probs_ema.mul_(self.rho_pi).add_(probs.mean(dim=0), alpha=self.rho_pi_c)
        return weights, selected, entropy.mean(), anti_sink, entropy

    @torch.no_grad()
    def _intervention_mask(self, anti_sink, entropy):
        """Fig. 2(b): confidence-matched sink-aligned (SA) / sink-repulsive (SR) halves."""
        mode = self.variant['intervention']
        if mode == 'all':
            return torch.ones_like(anti_sink, dtype=torch.bool)
        repulsive = anti_sink >= anti_sink.median()
        group = repulsive if mode == 'sr' else ~repulsive
        other = ~group
        edges = torch.quantile(entropy.float(), torch.linspace(0, 1, 11, device=entropy.device))
        bins = torch.bucketize(entropy.float(), edges[1:-1])
        mask = torch.zeros_like(group)
        gen = torch.Generator().manual_seed(self._batch_counter)
        for b in range(10):
            idx_g = (group & (bins == b)).nonzero().flatten()
            idx_o = (other & (bins == b)).nonzero().flatten()
            k = min(idx_g.numel(), idx_o.numel())
            if k > 0:
                pick = torch.randperm(idx_g.numel(), generator=gen)[:k].to(idx_g.device)
                mask[idx_g[pick]] = True
        return mask

    def _record_marginal(self):
        pi = self.class_probs_ema
        self._tr('hpi', -(pi * pi.clamp_min(EPS).log()).sum() / self._log_c)

    # ------------------------------------------------------------------ forward
    @torch.enable_grad()
    def forward(self, keypoint: torch.Tensor, y=None, adapt: bool = True):
        if not adapt:
            with torch.no_grad():
                return self.model(keypoint)
        self._batch_counter += 1
        orient = self.variant['orient']
        if self.variant['intervention'] is not None:
            return self._forward_intervention(keypoint)
        if orient == 'per_sample':
            return self._forward_per_sample(keypoint)
        if orient == 'none':
            return self._forward_fixed(keypoint, reverse=False)
        if orient == 'oracle':
            return self._forward_fixed(keypoint, reverse=self._stream_tag == REVERSE_TAG)

        # ---- 'vote' (original code path) and 'vote_raw'
        self.optimizer.zero_grad()
        reversed_keypoint = keypoint.flip([TIME_DIM])
        reverse = self._reverse_score > 0.0
        outputs = self.model(reversed_keypoint if reverse else keypoint)
        prediction = outputs.detach()
        weights, selected, entropy_mean, _, _ = self._sample_weights(prediction)

        with torch.no_grad():
            alternate = self.model(keypoint if reverse else reversed_keypoint)
            alt_entropy = shannon_entropy(alternate).mean()
            if orient == 'vote_raw':
                preference = (entropy_mean - alt_entropy).item()
            else:
                preference = (entropy_mean - alt_entropy).sign().item()
            self._reverse_score = self.rho_s * self._reverse_score + self.rho_s_c * (
                -preference if reverse else preference
            )

        self._step(outputs, selected, weights)
        self._tr('o', float(reverse))
        self._tr('s', self._reverse_score)
        self._tr('vote', -preference if reverse else preference)
        self._tr('h_can', entropy_mean)
        self._tr('h_alt', alt_entropy)
        self._tr('n_sel', selected.sum())
        self._record_marginal()
        return prediction

    def _step(self, outputs, selected, weights, denom=None):
        if self.variant.get('update', True) and selected.any():
            denom = outputs.shape[0] if denom is None else denom
            loss = (
                shannon_entropy(outputs[selected]) * weights[selected]
            ).sum() / denom
            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()
        self.optimizer.zero_grad()

    def _forward_fixed(self, keypoint, reverse):
        """No vote: 'none' (never reverse) or 'oracle' (reverse iff the domain is reversed)."""
        self.optimizer.zero_grad()
        outputs = self.model(keypoint.flip([TIME_DIM]) if reverse else keypoint)
        prediction = outputs.detach()
        weights, selected, entropy_mean, _, _ = self._sample_weights(prediction)
        self._step(outputs, selected, weights)
        self._tr('o', float(reverse))
        self._tr('h_can', entropy_mean)
        self._tr('n_sel', selected.sum())
        self._record_marginal()
        return prediction

    def _forward_per_sample(self, keypoint):
        """Tab. 4: every sample takes its own lower-entropy view (no stream-level vote)."""
        self.optimizer.zero_grad()
        reversed_keypoint = keypoint.flip([TIME_DIM])
        with torch.no_grad():
            h_fwd = shannon_entropy(self.model(keypoint))
            h_rev = shannon_entropy(self.model(reversed_keypoint))
            choose_rev = h_rev < h_fwd
            shape = [-1] + [1] * (keypoint.dim() - 1)
            canonical = torch.where(choose_rev.view(shape), reversed_keypoint, keypoint)
        outputs = self.model(canonical)
        prediction = outputs.detach()
        weights, selected, entropy_mean, _, _ = self._sample_weights(prediction)
        self._step(outputs, selected, weights)
        self._tr('o', choose_rev.float().mean())
        self._tr('h_can', entropy_mean)
        self._tr('n_sel', selected.sum())
        self._record_marginal()
        return prediction

    def _forward_intervention(self, keypoint):
        self.optimizer.zero_grad()
        outputs = self.model(keypoint)
        prediction = outputs.detach()
        _, _, entropy_mean, anti_sink, entropy = self._sample_weights(prediction)
        mask = self._intervention_mask(anti_sink, entropy)
        ones = torch.ones_like(anti_sink)
        if self.variant['intervention'] == 'all':
            self._step(outputs, mask, ones)                                   # mean over the batch
        else:
            self._step(outputs, mask, ones, denom=max(int(mask.sum()), 1))    # mean over the half
        self._tr('o', 0.0)
        self._tr('h_can', entropy_mean)
        self._tr('n_sel', mask.sum())
        self._record_marginal()
        return prediction


# ---------------------------------------------------------------------- set-up
def _configure_adaptation_model(model):
    from .TTAOps import configure_adaptation_model
    return configure_adaptation_model(model)


def _run_note(cfg):
    note = ''
    experiment = getattr(cfg, 'experiment', None)
    if experiment is not None:
        note = str(getattr(experiment, 'note', '') or '')
    return note or os.environ.get('RUN_NOTE', '') or 'full'


def setup(model, cfg):
    method_cfg = cfg.method.SRT_TTA
    num_classes = model.recognizer.cls_head.num_classes
    variant = parse_note(_run_note(cfg))
    source_bn = ([(m, m.running_mean.clone(), m.running_var.clone())
                  for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
                 if variant.get('source_bn', False) else [])
    model, params, _ = _configure_adaptation_model(model)
    for module, mean, variance in source_bn:
        module.running_mean, module.running_var = mean, variance
        module.track_running_stats = True
        module.eval()
    if source_bn:
        print('[SRT-TTA] Diagnostic: fixed source BN statistics; original affine SGD updates', flush=True)
    optimizer = torch.optim.SGD(
        params, lr=method_cfg.lr, momentum=method_cfg.momentum
    )
    adapter = SRT_TTA_paper(model, optimizer, num_classes, variant)
    info = {"SRT_TTA_paper_lr": method_cfg.lr, "SRT_TTA_paper_steps": 1}
    info.update({f"variant_{k}": v for k, v in variant.items()})
    return adapter, info
