import math
from copy import deepcopy

import torch
import torch.nn as nn

from .TTAOps import collect_bn_params, configure_tent_model, softmax_entropy


class ETAAdapter(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_margin=None):
        super().__init__()
        assert steps > 0
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.entropy_margin = entropy_margin
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

    def forward_and_adapt(self, x):
        outputs = self.model(x)
        entropies = softmax_entropy(outputs)
        if self.entropy_margin is not None:
            selected = entropies < self.entropy_margin
            if not torch.any(selected):
                return outputs.detach()
            loss = entropies[selected].mean()
        else:
            loss = entropies.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs


def setup(model, cfg):
    model = configure_tent_model(model)
    params, _ = collect_bn_params(model)
    optimizer = torch.optim.SGD(params, lr=cfg.method.eta.lr, momentum=cfg.method.eta.momentum)
    num_classes = getattr(cfg.dataset, 'num_classes', None)
    margin_scale = cfg.method.eta.margin_scale
    entropy_margin = None
    if num_classes:
        entropy_margin = math.log(num_classes) * margin_scale
    return ETAAdapter(
        model,
        optimizer,
        steps=cfg.method.eta.steps,
        episodic=cfg.method.eta.episodic,
        entropy_margin=entropy_margin,
    ), {
        'eta_lr': cfg.method.eta.lr,
        'eta_steps': cfg.method.eta.steps,
        'eta_margin_scale': margin_scale,
    }
