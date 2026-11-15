"""ProtoGCN adaptation of the official ZeroSiam asymmetric entropy objective.

Reference: https://github.com/Cascol-Chen/ZeroSiam (tta_library/zerosiam.py).
One encoder pass; identity linear predictor; entropy + symmetric KL to stop-grad
source-branch predictions. Normalization includes ProtoGCN's BatchNorm1d layers.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TTAOps import configure_adaptation_model, softmax_entropy


class ZeroSiamAdapter(nn.Module):
    def __init__(self, model, params, lr, momentum):
        super().__init__()
        self.model = model
        weight = model.recognizer.cls_head.fc_cls.weight
        self.predictor = nn.Linear(weight.shape[1], weight.shape[1], bias=False).to(weight)
        nn.init.eye_(self.predictor.weight)
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        self.predictor_optimizer = torch.optim.SGD(self.predictor.parameters(), lr=lr,
                                                   momentum=momentum)
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.predictor_optimizer_state = deepcopy(self.predictor_optimizer.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state)
        nn.init.eye_(self.predictor.weight)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.predictor_optimizer.load_state_dict(self.predictor_optimizer_state)

    @torch.enable_grad()
    def forward(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        self.predictor_optimizer.zero_grad(set_to_none=True)
        latent, _ = self.model.recognizer.extract_feat(x)
        features = latent.mean((1, 3, 4))
        head = self.model.recognizer.cls_head.fc_cls
        target = head(features).detach()
        online = head(self.predictor(features))
        alignment = (F.kl_div(online.log_softmax(1), target.softmax(1), reduction='none') +
                     F.kl_div(target.log_softmax(1), online.softmax(1), reduction='none')).sum(1)
        loss = (softmax_entropy(online) + alignment).mean()
        loss.backward()
        self.optimizer.step()
        self.predictor_optimizer.step()
        return target


def setup(model, cfg):
    model, params, _ = configure_adaptation_model(model)
    options = cfg.method.zerosiam
    return ZeroSiamAdapter(model, params, options.lr, options.momentum), vars(options)
