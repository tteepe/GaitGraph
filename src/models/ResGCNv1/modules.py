import logging, torch
from torch import nn


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class ResGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, block, A, initial=False, stride=1, kernel_size=[9,2], **kwargs):
        super(ResGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = import_class('models.ResGCNv1.blocks.Spatial_{}_Block'.format(block))
        temporal_block = import_class('models.ResGCNv1.blocks.Temporal_{}_Block'.format(block))
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        return self.tcn(self.scn(x, A*self.edge), self.residual(x))


class AttGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, block, A, attention, stride=1, kernel_size=[9,2], **kwargs):
        super(AttGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = import_class('models.ResGCNv1.blocks.Spatial_{}_Block'.format(block))
        temporal_block = import_class('models.ResGCNv1.blocks.Temporal_{}_Block'.format(block))
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.att = attention(out_channels, **kwargs)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        return self.att(self.tcn(self.scn(x, A*self.edge), self.residual(x)))
