import torch

from copy import deepcopy

import torch
import torch.nn as nn


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class BNModel(nn.Module):
    def __init__(self, model, steps=1):
        super().__init__()
        self.steps = steps
        assert steps > 0, 'BN requires >= 1 step(s) to forward and update'
        self.model = model
        self.model_state = deepcopy(self.model.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def forward(self, x):
        return self.model(x)


class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        assert steps > 0
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
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
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs


def configure_bn_model(model, eps, momentum, reset_stats, no_stats):
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.requires_grad_(False)
            module.eps = eps
            module.momentum = momentum
            if reset_stats:
                module.reset_running_stats()
            if no_stats:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            else:
                module.track_running_stats = True
            module.train()
    return model


def configure_tent_model(model):
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def configure_adaptation_model(model, skip_keywords=None):
    skip_keywords = tuple(skip_keywords or ())
    model.eval()
    model.requires_grad_(False)
    params = []
    names = []
    for module_name, module in model.named_modules():
        if skip_keywords and any(token in module_name for token in skip_keywords):
            continue
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ['weight', 'bias']:
                    param.requires_grad_(True)
                    params.append(param)
                    names.append(f'{module_name}.{param_name}')
    return model, params, names


def collect_bn_params(model):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param_name, param in module.named_parameters():
                if param_name in ['weight', 'bias']:
                    params.append(param)
                    names.append(f'{module_name}.{param_name}')
    return params, names


def setup(model, cfg):
    model = configure_tent_model(model)
    params, _ = collect_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.method.tent.lr)
    return Tent(model, optimizer, steps=cfg.method.tent.steps, episodic=cfg.method.tent.episodic), {
        'tent_lr': cfg.method.tent.lr,
        'tent_steps': cfg.method.tent.steps,
    }
