from collections import deque

import torch


class RoTTAMemory:
    def __init__(self, capacity, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.entries = deque(maxlen=capacity)

    def clear(self):
        self.entries.clear()

    def add(self, pseudo_probs, uncertainty):
        for prob, unc in zip(pseudo_probs.detach(), uncertainty.detach()):
            self.entries.append({
                'prob': prob.clone(),
                'age': 0.0,
                'uncertainty': float(unc.item()),
            })
        self.increment_ages(skip_recent=len(pseudo_probs))

    def increment_ages(self, skip_recent=0):
        cutoff = max(0, len(self.entries) - skip_recent)
        for i, entry in enumerate(self.entries):
            if i < cutoff:
                entry['age'] += 1.0

    def ready(self, update_frequency):
        return len(self.entries) >= max(1, update_frequency)

    def get_weighted_probs(self):
        if not self.entries:
            return None
        probs = torch.stack([entry['prob'] for entry in self.entries], dim=0)
        ages = torch.tensor([entry['age'] for entry in self.entries], device=probs.device, dtype=probs.dtype)
        uncertainties = torch.tensor([entry['uncertainty'] for entry in self.entries], device=probs.device, dtype=probs.dtype)
        age_weights = torch.exp(-self.lambda_t * ages)
        unc_weights = torch.exp(-self.lambda_u * uncertainties)
        weights = age_weights * unc_weights
        weights = weights / weights.sum().clamp_min(1e-6)
        return (weights.unsqueeze(1) * probs).sum(0), weights
