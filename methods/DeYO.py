import math
from copy import deepcopy

import torch
import torch.nn as nn

from .TTAOps import configure_adaptation_model, softmax_entropy


class DeYOAdapter(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_margin=None, margin_e0=None, plpd_threshold=0.0, reweight_ent=True, reweight_plpd=True, aug_type='shuffle', noise_std=0.01):
        super().__init__()
        assert steps > 0
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.entropy_margin = entropy_margin
        self.margin_e0 = margin_e0 if margin_e0 is not None else entropy_margin
        self.plpd_threshold = plpd_threshold
        self.reweight_ent = reweight_ent
        self.reweight_plpd = reweight_plpd
        self.aug_type = aug_type
        self.noise_std = noise_std
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def _augment(self, x):
        if self.aug_type == 'noise':
            return x + torch.randn_like(x) * self.noise_std
        if self.aug_type == 'temporal_flip' and x.dim() >= 3:
            return torch.flip(x, dims=[-1])
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        idx = torch.randperm(flat.shape[-1], device=flat.device)
        shuffled = flat[:, :, idx]
        return shuffled.reshape_as(x)

    def forward_and_adapt(self, x):
        outputs = self.model(x)
        entropies = softmax_entropy(outputs)
        keep = torch.ones_like(entropies, dtype=torch.bool)
        if self.entropy_margin is not None:
            keep &= entropies < self.entropy_margin
        if not torch.any(keep):
            return outputs.detach()

        x_prime = self._augment(x[keep].detach().clone())
        with torch.no_grad():
            outputs_prime = self.model(x_prime)
        probs = outputs[keep].softmax(1)
        probs_prime = outputs_prime.softmax(1)
        cls = probs.argmax(dim=1, keepdim=True)
        plpd = torch.gather(probs, 1, cls) - torch.gather(probs_prime, 1, cls)
        select = plpd.squeeze(1) > self.plpd_threshold
        if not torch.any(select):
            return outputs.detach()

        chosen_entropy = entropies[keep][select]
        chosen_plpd = plpd.squeeze(1)[select]
        coeff = torch.ones_like(chosen_entropy)
        if self.reweight_ent and self.margin_e0 is not None:
            coeff = coeff + 1.0 / torch.exp(chosen_entropy.detach() - self.margin_e0)
        if self.reweight_plpd:
            coeff = coeff + 1.0 / torch.exp(-chosen_plpd.detach())
        loss = (chosen_entropy * coeff).mean()
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return outputs.detach()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs


def setup(model, cfg):
    skip_keywords = list(getattr(cfg.method.deyo, 'skip_keywords', []))
    model, params, _ = configure_adaptation_model(model, skip_keywords=skip_keywords)
    optimizer = torch.optim.Adam(params, lr=cfg.method.deyo.lr)
    num_classes = getattr(cfg.dataset, 'num_classes', None)
    entropy_margin = None
    margin_e0 = None
    if num_classes:
        entropy_margin = math.log(num_classes) * cfg.method.deyo.margin_scale
        margin_e0 = math.log(num_classes) * cfg.method.deyo.margin_e0_scale
    return DeYOAdapter(
        model,
        optimizer,
        steps=cfg.method.deyo.steps,
        episodic=cfg.method.deyo.episodic,
        entropy_margin=entropy_margin,
        margin_e0=margin_e0,
        plpd_threshold=cfg.method.deyo.plpd_threshold,
        reweight_ent=cfg.method.deyo.reweight_ent,
        reweight_plpd=cfg.method.deyo.reweight_plpd,
        aug_type=cfg.method.deyo.aug_type,
        noise_std=cfg.method.deyo.noise_std,
    ), {
        'deyo_lr': cfg.method.deyo.lr,
        'deyo_steps': cfg.method.deyo.steps,
        'deyo_plpd_threshold': cfg.method.deyo.plpd_threshold,
        'deyo_aug_type': cfg.method.deyo.aug_type,
    }
