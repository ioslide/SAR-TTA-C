from copy import deepcopy

import torch
import torch.nn as nn


class RecognizerWrapper(nn.Module):
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, keypoint):
        return self.forward_full(keypoint)[0]

    def forward_full(self, keypoint):
        x, graph = self.recognizer.extract_feat(keypoint)
        return self.recognizer.cls_head(x), graph


class SourceAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.model_state = deepcopy(self.model.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
