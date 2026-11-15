"""ETA with periodic source resets, following oripress/CCC/models/rdumb.py."""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TTAOps import configure_adaptation_model, softmax_entropy


class RDumbAdapter(nn.Module):
    def __init__(self, model, optimizer, reset_interval=1000):
        super().__init__()
        self.model, self.optimizer = model, optimizer
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self.reset_interval = reset_interval
        self.entropy_margin = 0.4 * math.log(model.recognizer.cls_head.num_classes)
        self.current_model_probs = None
        self.total_steps = 0

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.current_model_probs = None
        self.total_steps = 0

    @torch.enable_grad()
    def forward(self, x):
        if self.total_steps and self.total_steps % self.reset_interval == 0:
            self.model.load_state_dict(self.model_state)
            self.optimizer.load_state_dict(self.optimizer_state)
            self.current_model_probs = None
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        entropy = softmax_entropy(logits)
        keep = entropy < self.entropy_margin
        if self.current_model_probs is not None:
            similarity = F.cosine_similarity(logits.softmax(1).detach(),
                                             self.current_model_probs[None], dim=1)
            keep &= similarity.abs() < 0.05
        if keep.any():
            probs = logits[keep].softmax(1).detach().mean(0)
            self.current_model_probs = (probs if self.current_model_probs is None else
                                        0.9 * self.current_model_probs + 0.1 * probs)
            loss = (entropy[keep] * (self.entropy_margin - entropy[keep].detach()).exp()).mean()
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.total_steps += 1
        return logits.detach()


def setup(model, cfg):
    model, params, _ = configure_adaptation_model(model)
    options = cfg.method.rdumb
    optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum)
    return RDumbAdapter(model, optimizer, options.reset_interval), vars(options)
