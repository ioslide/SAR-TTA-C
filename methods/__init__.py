from .Source import setup as setup_0
from .BN import setup as setup_1
from .Tent import setup as setup_2
from .ETA import setup as setup_3
from .EATA import setup as setup_4
from .CoTTA import setup as setup_5
from .RoTTA import setup as setup_6
from .RDumb import setup as setup_7
from .DeYO import setup as setup_8
from .NEO import setup as setup_9
from .ZeroSiam import setup as setup_10
from .SkeletonCacheUniform import setup as setup_11
from .SRT_TTA_paper import setup as setup_12
from .PaperExtensions import setup_group, setup_srem, setup_srem_im, setup_tent_im

METHOD_REGISTRY = {
    'Source': setup_0,
    'BN': setup_1,
    'Tent': setup_2,
    'ETA': setup_3,
    'EATA': setup_4,
    'CoTTA': setup_5,
    'RoTTA': setup_6,
    'RDumb': setup_7,
    'DeYO': setup_8,
    'NEO': setup_9,
    'ZeroSiam': setup_10,
    'SkeletonCacheUniform': setup_11,
    'SRT_TTA_paper': setup_12,
    'SRT': setup_12,
    'Skeleton-Cache': setup_11,
    'SRT_G': setup_group,
    'SREM': setup_srem,
    'SREM_IM': setup_srem_im,
    'Tent_IM': setup_tent_im,
}

def setup_method(name, model, cfg):
    return METHOD_REGISTRY[name](model, cfg)
