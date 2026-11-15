from copy import deepcopy

import torch.nn as nn

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class AdaBN(nn.Module):
    def __init__(self, model, eps=1e-5, momentum=0.1, reset_stats=False, no_stats=False):
        super().__init__()
        self.model = configure_model(model, eps, momentum, reset_stats, no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def forward(self, x):
        return self.model(x)


def configure_model(model, eps=1e-5, momentum=0.1, reset_stats=False, no_stats=False):
    for m in model.modules():
        if isinstance(m, BN_TYPES):
            m.train()
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                m.reset_running_stats()
            if no_stats:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
