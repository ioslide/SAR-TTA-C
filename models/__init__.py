import sys
from pathlib import Path

from mmcv.runner import load_checkpoint

from protogcn.models import build_model as build_protogcn_model
from methods.Common import RecognizerWrapper


class PysklRecognizerWrapper(RecognizerWrapper):
    def forward_full(self, keypoint):
        features = self.recognizer.extract_feat(keypoint)
        return self.recognizer.cls_head(features), None


def build_model(model_cfg, checkpoint=None):
    wrapper = RecognizerWrapper
    if model_cfg['backbone']['type'] in ('STGCN', 'CTRGCN'):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'pyskl'))
        from pyskl.models import build_model as build_pyskl_model
        model = build_pyskl_model(model_cfg)
        wrapper = PysklRecognizerWrapper
    else:
        model = build_protogcn_model(model_cfg)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
    return wrapper(model).cuda()
