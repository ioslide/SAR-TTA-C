from copy import deepcopy

import torch
import torch.nn as nn

from .RoTTAMemory import RoTTAMemory
from .TTAOps import collect_bn_params, configure_tent_model, softmax_entropy


class RoTTAAdapter(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, memory_size=64, ema_momentum=0.99, update_frequency=16, lambda_t=1.0, lambda_u=1.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.memory_size = memory_size
        self.ema_momentum = ema_momentum
        self.update_frequency = update_frequency
        self.teacher = deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.memory = RoTTAMemory(capacity=memory_size, lambda_t=lambda_t, lambda_u=lambda_u)
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.teacher_state = deepcopy(self.teacher.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.teacher.load_state_dict(self.teacher_state, strict=True)
        self.memory.clear()

    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def forward_and_adapt(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
            pseudo_probs = teacher_logits.softmax(1)
            uncertainty = softmax_entropy(teacher_logits)
            self.memory.add(pseudo_probs, uncertainty)
        if not self.memory.ready(self.update_frequency):
            return teacher_logits

        student_logits = self.model(x)
        weighted_target, weights = self.memory.get_weighted_probs()
        target = weighted_target.unsqueeze(0).expand_as(student_logits)
        entropy_loss = softmax_entropy(student_logits).mean()
        consistency_loss = -(target * student_logits.log_softmax(1)).sum(1).mean()
        loss = consistency_loss + 0.1 * entropy_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
                teacher_param.data.mul_(self.ema_momentum).add_(student_param.data, alpha=1 - self.ema_momentum)
        return teacher_logits


def setup(model, cfg):
    model = configure_tent_model(model)
    params, _ = collect_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.method.rotta.lr)
    return RoTTAAdapter(
        model,
        optimizer,
        steps=cfg.method.rotta.steps,
        episodic=cfg.method.rotta.episodic,
        memory_size=cfg.method.rotta.memory_size,
        ema_momentum=cfg.method.rotta.ema_momentum,
        update_frequency=cfg.method.rotta.update_frequency,
        lambda_t=cfg.method.rotta.lambda_t,
        lambda_u=cfg.method.rotta.lambda_u,
    ), {
        'rotta_lr': cfg.method.rotta.lr,
        'rotta_steps': cfg.method.rotta.steps,
        'rotta_memory_size': cfg.method.rotta.memory_size,
        'rotta_update_frequency': cfg.method.rotta.update_frequency,
    }
