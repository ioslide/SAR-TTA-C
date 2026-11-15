_base_ = './stgcnpp_tta_bs32.py'

model = dict(
    _delete_=True,
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
