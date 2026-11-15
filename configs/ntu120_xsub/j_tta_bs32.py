_base_ = '../ntu60_xview/j_tta_bs32.py'

model = dict(cls_head=dict(num_classes=120))
ann_file = 'data/nturgbd/ntu120_3danno.pkl'
work_dir = './work_dirs/ntu120_xsub/j_tta_bs32'
data = dict(
    train=dict(ann_file=ann_file, split='xsub_train'),
    val=dict(ann_file=ann_file, split='xsub_val'),
    test=dict(ann_file=ann_file, split='xsub_val'))
