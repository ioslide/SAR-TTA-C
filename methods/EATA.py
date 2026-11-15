import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TTAOps import collect_bn_params, configure_tent_model, softmax_entropy


class EATAAdapter(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_margin=None, redundancy_margin=0.05):
        super().__init__()
        assert steps > 0
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.entropy_margin = entropy_margin
        self.redundancy_margin = redundancy_margin
        self.current_model_probs = None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.current_model_probs = None

    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def forward_and_adapt(self, x):
        outputs = self.model(x)
        entropies = softmax_entropy(outputs)
        keep = torch.ones_like(entropies, dtype=torch.bool)
        if self.entropy_margin is not None:
            keep &= entropies < self.entropy_margin
        probs = outputs.softmax(1)
        if self.current_model_probs is not None and torch.any(keep):
            similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(0), probs, dim=1)
            keep &= torch.abs(similarities) < self.redundancy_margin
        if not torch.any(keep):
            return outputs.detach()
        loss = entropies[keep].mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            batch_mean = probs[keep].mean(0)
            if self.current_model_probs is None:
                self.current_model_probs = batch_mean
            else:
                self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * batch_mean
        return outputs


def setup(model, cfg):
    model = configure_tent_model(model)
    params, _ = collect_bn_params(model)
    optimizer = torch.optim.SGD(params, lr=cfg.method.eata.lr, momentum=cfg.method.eata.momentum)
    num_classes = getattr(cfg.dataset, 'num_classes', None)
    margin_scale = cfg.method.eata.margin_scale
    entropy_margin = None
    if num_classes:
        entropy_margin = math.log(num_classes) * margin_scale
    return EATAAdapter(
        model,
        optimizer,
        steps=cfg.method.eata.steps,
        episodic=cfg.method.eata.episodic,
        entropy_margin=entropy_margin,
        redundancy_margin=cfg.method.eata.redundancy_margin,
    ), {
        'eata_lr': cfg.method.eata.lr,
        'eata_steps': cfg.method.eata.steps,
        'eata_margin_scale': margin_scale,
        'eata_redundancy_margin': cfg.method.eata.redundancy_margin,
    }
