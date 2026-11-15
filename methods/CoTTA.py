from copy import deepcopy

import torch
import torch.nn as nn

from .TTAOps import collect_bn_params, configure_tent_model


class CoTTAAdapter(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, ema_momentum=0.999, restore_prob=0.01, aug_views=4, noise_std=0.01):
        super().__init__()
        self.student = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.ema_momentum = ema_momentum
        self.restore_prob = restore_prob
        self.aug_views = aug_views
        self.noise_std = noise_std
        self.teacher = deepcopy(model)
        self.anchor = deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.model_state = deepcopy(self.student.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.teacher_state = deepcopy(self.teacher.state_dict())

    def reset(self):
        self.student.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.teacher.load_state_dict(self.teacher_state, strict=True)

    def _augment(self, x):
        noise = torch.randn_like(x) * self.noise_std
        if x.dim() >= 3:
            flip = torch.rand(1, device=x.device).item() > 0.5
            aug = x + noise
            if flip:
                aug = torch.flip(aug, dims=[-1])
            return aug
        return x + noise

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
            aug_logits = [self.teacher(self._augment(x)) for _ in range(max(1, self.aug_views - 1))]
            mean_teacher = torch.stack([teacher_logits] + aug_logits, dim=0).mean(0)
            pseudo = mean_teacher.softmax(1)

        student_logits = self.student(x)
        student_log_probs = student_logits.log_softmax(1)
        loss = -(pseudo * student_log_probs).sum(1).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data.mul_(self.ema_momentum).add_(student_param.data, alpha=1 - self.ema_momentum)
            for student_param, anchor_param in zip(self.student.parameters(), self.anchor.parameters()):
                mask = torch.rand_like(student_param) < self.restore_prob
                student_param.data = torch.where(mask, anchor_param.data, student_param.data)
        return mean_teacher


def setup(model, cfg):
    model = configure_tent_model(model)
    params, _ = collect_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.method.cotta.lr)
    return CoTTAAdapter(
        model,
        optimizer,
        steps=cfg.method.cotta.steps,
        episodic=cfg.method.cotta.episodic,
        ema_momentum=cfg.method.cotta.ema_momentum,
        restore_prob=cfg.method.cotta.restore_prob,
        aug_views=cfg.method.cotta.aug_views,
        noise_std=cfg.method.cotta.noise_std,
    ), {
        'cotta_lr': cfg.method.cotta.lr,
        'cotta_steps': cfg.method.cotta.steps,
        'cotta_ema_momentum': cfg.method.cotta.ema_momentum,
        'cotta_restore_prob': cfg.method.cotta.restore_prob,
    }
