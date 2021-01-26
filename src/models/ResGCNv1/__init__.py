import logging

from . import blocks
from .nets import ResGCN
from .modules import ResGCN_Module, AttGCN_Module
from .attentions import *


__model = {
    'resgcn': ResGCN,
}

__attention = {
    'pa': Part_Att,
    'ca': Channel_Att,
    'fa': Frame_Att,
    'ja': Joint_Att,
    'pca': Part_Conv_Att,
    'psa': Part_Share_Att,
}

__structure = {
    'b15': {'structure': [1,2,2,2], 'block': 'Basic'},
    'b19': {'structure': [1,2,3,3], 'block': 'Basic'},
    'b23': {'structure': [1,3,4,3], 'block': 'Basic'},
    'b29': {'structure': [1,3,6,4], 'block': 'Basic'},
    'n39': {'structure': [1,2,2,2], 'block': 'Bottleneck'},
    'n51': {'structure': [1,2,3,3], 'block': 'Bottleneck'},
    'n57': {'structure': [1,3,4,3], 'block': 'Bottleneck'},
    'n75': {'structure': [1,3,6,4], 'block': 'Bottleneck'},
}

__reduction = {
    'r1': {'reduction': 1},
    'r2': {'reduction': 2},
    'r4': {'reduction': 4},
    'r8': {'reduction': 8},
}


def create(model_type, **kwargs):
    model_split = model_type.split('-')
    if model_split[0] in __attention.keys():
        kwargs.update({'module': AttGCN_Module, 'attention': __attention[model_split[0]]})
        del(model_split[0])
    else:
        kwargs.update({'module': ResGCN_Module, 'attention': None})
    try:
        [model, structure, reduction] = model_split
    except:
        [model, structure], reduction = model_split, 'r1'
    if not (model in __model.keys() and structure in __structure.keys() and reduction in __reduction.keys()):
        logging.info('')
        logging.error('Error: Do NOT exist this model_type: {}!'.format(model_type))
        raise ValueError()
    return __model[model](**(__structure[structure]), **(__reduction[reduction]), **kwargs)
