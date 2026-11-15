"""Causal ProtoGCN port of NEO's continual latent recentering (alpha=0.1).

The centroid updates AFTER prediction to retain this benchmark's online protocol.
Reference: https://github.com/awesomealex1/NEO/blob/main/neo_layers.py
"""
import torch
import torch.nn as nn


class NEOAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.register_buffer('center', torch.zeros_like(
            model.recognizer.cls_head.fc_cls.weight[0]))

    def reset(self):
        self.center.zero_()

    @torch.no_grad()
    def forward(self, x):
        latent, _ = self.model.recognizer.extract_feat(x)
        features = latent.mean((1, 3, 4))
        logits = self.model.recognizer.cls_head.fc_cls(features - self.center)
        self.center.mul_(0.9).add_(features.mean(0), alpha=0.1)
        return logits


def setup(model, cfg):
    print('NEO: ProtoGCN continual port; centroid EMA=0.1, predict before update')
    return NEOAdapter(model), {'centroid_rate': 0.1, 'causal': True}
