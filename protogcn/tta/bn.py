from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ['BNModel', 'configure_model', 'setup']


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


def configure_model(model, eps, momentum, reset_stats, no_stats):
    # Put the whole model in eval mode first; then selectively switch BN layers to train.
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(False)
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                m.reset_running_stats()
            if no_stats:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.track_running_stats = True
            m.train()
    return model


def setup(model, cfg):
    model = configure_model(
        model,
        cfg.ADAPTER.BN.EPS,
        cfg.OPTIM.MOMENTUM,
        cfg.ADAPTER.BN.RESET_STATS,
        cfg.ADAPTER.BN.NO_STATS,
    )
    return BNModel(model, steps=int(cfg.OPTIM.STEPS))
