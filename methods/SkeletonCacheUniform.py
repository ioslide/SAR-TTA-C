"""Causal, closed-set Skeleton-Cache port with uniform descriptor fusion.

Descriptor groups/retrieval follow Alchemist0754/Skeleton-Cache (BSD-3-Clause;
see SkeletonCache.LICENSE). This is not the original zero-shot model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassFeatureBank(nn.Module):
    def __init__(self, reference, capacity, feature_shape):
        super().__init__()
        self.register_buffer('keys', reference.new_zeros(
            reference.shape[0], capacity, *feature_shape))
        self.register_buffer('quality', reference.new_full(
            (reference.shape[0], capacity), -torch.inf))

    @torch.no_grad()
    def push(self, keys, labels, quality):
        for key, label, score in zip(keys, labels.tolist(), quality):
            slot = self.quality[label].argmin()
            if score > self.quality[label, slot]:
                self.keys[label, slot].copy_(key)
                self.quality[label, slot] = score

    def items(self):
        valid = self.quality.isfinite()
        labels = torch.arange(self.keys.shape[0], device=self.keys.device)
        labels = labels[:, None].expand_as(valid)
        return self.keys[valid], labels[valid]

    @torch.no_grad()
    def reset(self):
        self.keys.zero_()
        self.quality.fill_(-torch.inf)


def structured_descriptors(features):
    groups = (
        (2, 3, 4, 8, 20),
        (0, 1, 4, 8, 12, 16, 20),
        (4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24),
        (0, 12, 13, 14, 15, 16, 17, 18, 19),
    )
    descriptors = [features[..., joints].mean((-1, -2)) for joints in groups]
    length = features.shape[2]
    edges = (0, round(length / 3), round(2 * length / 3), length)
    descriptors += [features[:, :, start:end].mean((-1, -2))
                    for start, end in zip(edges[:-1], edges[1:])]
    descriptors.append(features.mean((-1, -2)))
    return F.normalize(torch.stack(descriptors, dim=1), dim=-1)


class SkeletonCacheUniformAdapter(nn.Module):
    def __init__(self, model, capacity=8, alpha=1.0, beta=5.0):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        weight = model.recognizer.cls_head.fc_cls.weight
        self.bank = ClassFeatureBank(weight, capacity, (8, weight.shape[1]))
        self.alpha = alpha
        self.beta = beta

    def reset(self):
        self.bank.reset()

    @torch.no_grad()
    def forward(self, x):
        latent, _ = self.model.recognizer.extract_feat(x)
        logits = self.model.recognizer.cls_head(latent)
        query = structured_descriptors(latent.mean(1))
        keys, labels = self.bank.items()
        if labels.numel():
            cosine = torch.einsum('bsd,nsd->bsn', query, keys)
            affinity = torch.exp(-self.beta * (1 - cosine)).mean(1)
            scores = affinity @ F.one_hot(labels, logits.shape[1]).to(logits.dtype)
            logits = logits + self.alpha * scores
        log_probs = logits.log_softmax(1)
        quality = (log_probs.exp() * log_probs).sum(1)
        # Predict with the previous bank, then store the current batch.
        self.bank.push(query, logits.argmax(1), quality)
        return logits


def setup(model, cfg):
    params = vars(cfg.method.skeleton_cache)
    print('SkeletonCacheUniform: closed-set ProtoGCN port; uniform fusion, causal cache')
    return SkeletonCacheUniformAdapter(model, **params), params
