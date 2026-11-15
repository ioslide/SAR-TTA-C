"""Additional experiments specified by the current paper; shared defaults unchanged."""
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .SRT_TTA_paper import (SRT_TTA_paper, _configure_adaptation_model,
                            min_max_scale, shannon_entropy)
from .Tent import Tent, setup as setup_tent
from paper_tools.variants import DEFAULT


def roll(keypoint, quarter_turns):
    """Exact rotations about the camera/depth z axis; zero joints remain zero."""
    turns = quarter_turns % 4
    if turns == 0:
        return keypoint
    output = keypoint.clone()
    x, y = keypoint[..., 0], keypoint[..., 1]
    if turns == 1:
        output[..., 0], output[..., 1] = -y, x
    elif turns == 2:
        output[..., 0], output[..., 1] = -x, -y
    else:
        output[..., 0], output[..., 1] = y, -x
    return output


class GroupSRT(SRT_TTA_paper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('group_state', self.class_probs_ema.new_zeros(8))
        self.group_state[0] = 1

    def reset(self):
        super().reset()
        self.group_state.zero_()
        self.group_state[0] = 1

    @staticmethod
    def canonicalize(x, element):
        output = roll(x, -(element % 4))
        return output.flip([2]) if element // 4 else output

    @torch.enable_grad()
    def forward(self, x):
        self.optimizer.zero_grad()
        current = int(self.group_state.argmax())
        outputs = self.model(self.canonicalize(x, current))
        prediction = outputs.detach()
        weights, selected, entropy, _, _ = self._sample_weights(prediction)
        with torch.no_grad():
            entropies = torch.stack([
                entropy if i == current else
                shannon_entropy(self.model(self.canonicalize(x, i))).mean()
                for i in range(8)])
            vote = int(entropies.argmin())
            self.group_state.mul_(self.rho_s)
            self.group_state[vote] += self.rho_s_c
        self._step(outputs, selected, weights)
        self._tr('group', current)
        self._tr('group_vote', vote)
        self._record_marginal()
        return prediction


class SREMWithIM(SRT_TTA_paper):
    def _step(self, outputs, selected, weights, denom=None):
        entropy_loss = (shannon_entropy(outputs) * weights).mean()
        marginal = outputs.softmax(1).mean(0)
        loss = entropy_loss + (marginal * marginal.clamp_min(1e-12).log()).sum()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class TentWithIM(Tent):
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()
        outputs = self.model(x)
        marginal = outputs.softmax(1).mean(0)
        loss = shannon_entropy(outputs).mean() + (marginal * marginal.clamp_min(1e-12).log()).sum()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs.detach()


def make_srt(model, cfg, cls=SRT_TTA_paper, **variant):
    model, params, _ = _configure_adaptation_model(model)
    options = cfg.method.SRT_TTA
    optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum)
    settings = dict(DEFAULT, **variant)
    return cls(model, optimizer, model.recognizer.cls_head.num_classes, settings)


def setup_group(model, cfg):
    return make_srt(model, cfg, GroupSRT), {'group': 'C4 x time-reversal', 'causal': True}


def setup_srem(model, cfg):
    return make_srt(model, cfg, orient='none'), {'ATC': False}


def setup_srem_im(model, cfg):
    return make_srt(model, cfg, SREMWithIM, orient='none'), {'IM_weight': 1.0, 'ATC': False}


def setup_tent_im(model, cfg):
    base, info = setup_tent(model, cfg)
    return TentWithIM(base.model, base.optimizer, base.steps, base.episodic), dict(info, IM_weight=1.0)


def entropy_gradient(probabilities):
    logp = probabilities.clamp_min(1e-12).log()
    entropy = -(probabilities * logp).sum(-1, keepdim=True)
    return -probabilities * (logp + entropy)


def gradient_dot(left, right):
    return sum((a * b).sum() for a, b in zip(left, right) if a is not None and b is not None)


def parameter_decomposition(outputs, params, update, half, d):
    """Decompose the receiving entropy change along -grad_theta L.

    JVPs include the shared BN statistics. Eq. (exact) applies to the mean
    logit push; nonshared_push records the additional per-sequence component.
    """
    receiving = outputs[half:]
    probe = torch.zeros_like(receiving, requires_grad=True)
    vjp = torch.autograd.grad(receiving, params, probe, create_graph=True,
                              retain_graph=True, allow_unused=True)
    direction = [None if g is None else -g.detach() for g in update]
    push = torch.autograd.grad(gradient_dot(vjp, direction), probe, retain_graph=True)[0]
    with torch.no_grad():
        p = receiving.softmax(1).double()
        push = push.double()
        mean, shared = p.mean(0), push.mean(0)
        log_mean = mean.clamp_min(1e-12).log()
        a, b = p @ log_mean, p @ shared
        between = ((a - a.mean()) * (b - b.mean())).mean()
        first = (entropy_gradient(mean) * shared).sum()
        residual = push - shared
        nonshared = -((p * log_mean * residual).sum(1)
                      - a * (p * residual).sum(1)).mean()
        return dict(parameter_first=float(first), parameter_between=float(between),
                    parameter_nonshared_push=float(nonshared),
                    parameter_decomposition_error=float(first + between + nonshared - d))


class TheorySRT(SRT_TTA_paper):
    """Record analytic logit decomposition and cross-fitted parameter-gradient D.

    Predictions and the actual update keep the full B=32 batch. For D estimation,
    the first half defines the update loss and the other half the receiving
    marginal. Both share the same forward/BN batch statistics. The optional exact
    gate compares B per-sample parameter gradients to the receiving gradient.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exact_gate = False
        self.decomposition = False
        self.measurements = []

    def _sample_weights(self, logits):
        self.prior = self.class_probs_ema.clone()
        return super()._sample_weights(logits)

    def _step(self, outputs, selected, weights, denom=None):
        # A final singleton batch has no disjoint halves; exclude it from D only.
        if len(outputs) < 2:
            return super()._step(outputs, selected, weights, denom)
        half = len(outputs) // 2
        params = [p for p in self.model.parameters() if p.requires_grad]
        probs = outputs.softmax(1)
        receiver = probs[half:].mean(0)
        f = -(receiver * receiver.clamp_min(1e-12).log()).sum()
        reference = torch.autograd.grad(f, params, retain_graph=True, allow_unused=True)
        entropies = shannon_entropy(outputs)
        if self.exact_gate:
            alignments = torch.stack([gradient_dot(
                torch.autograd.grad(value, params, retain_graph=True, allow_unused=True), reference)
                for value in entropies]).detach()
            anti = min_max_scale(-alignments)
            confidence = min_max_scale(-entropies.detach())
            selected.copy_(anti >= anti.mean())
            weights = (self.beta * anti * confidence).exp().masked_fill(~selected, 0)
        loss_half = (entropies[:half] * weights[:half]).mean()
        update = torch.autograd.grad(loss_half, params, retain_graph=True, allow_unused=True)
        d = -gradient_dot(reference, update).detach()
        with torch.no_grad():
            p = probs.detach()
            g = entropy_gradient(p)
            affinity = F.cosine_similarity(p.pow(self.alpha / 2),
                                           self.prior[None].pow(self.alpha / 2), dim=1)
            alignment = (g * entropy_gradient(self.prior)).sum(1)
            if affinity.max() > affinity.min() and alignment.max() > alignment.min():
                rho = float(spearmanr(affinity.cpu().numpy(), alignment.cpu().numpy()).correlation)
            else:
                rho = None
            delta = -(weights[:half, None] * g[:half]).mean(0)
            receiving = p[half:]
            mean = receiving.mean(0)
            a = receiving @ mean.clamp_min(1e-12).log()
            b = receiving @ delta
            between = ((a - a.mean()) * (b - b.mean())).mean()
            first = (entropy_gradient(mean) * delta).sum()
            total = first + between
            self.measurements.append(dict(spearman=rho, D=float(d),
                                           shared_push_D=float(total),
                                           between=float(between), first=float(first)))
        if self.decomposition:
            self.measurements[-1].update(parameter_decomposition(outputs, params, update, half, d))
        super()._step(outputs, selected, weights, denom)


def setup_theory(model, cfg, tent=False, exact=False, decomposition=False):
    if tent:
        base, _ = setup_tent(model, cfg)
        adapter = TheorySRT(base.model, base.optimizer, model.recognizer.cls_head.num_classes,
                            dict(DEFAULT, gate=False, weight='none', orient='none'))
    else:
        adapter = make_srt(model, cfg, TheorySRT)
    adapter.exact_gate = exact
    adapter.decomposition = decomposition
    return adapter
