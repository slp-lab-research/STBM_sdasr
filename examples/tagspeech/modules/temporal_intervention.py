"""In-memory inference interventions on a freshly loaded temporal conditioner."""
import torch
from torch import nn


def apply_temporal_intervention(conditioner, mode):
    if mode not in {'normal', 'bypass', 'center_only'}:
        raise ValueError(f'Unknown temporal intervention: {mode}')
    if conditioner.training:
        raise ValueError('Temporal interventions require eval mode')
    if hasattr(conditioner, '_applied_temporal_intervention'):
        raise ValueError('Load a fresh checkpoint for each intervention')
    if not conditioner.use_temporal_convolution or conditioner.temporal_convolution is None:
        raise ValueError('The source model must have an enabled temporal block')
    convs=[m for m in conditioner.temporal_convolution.modules() if isinstance(m,nn.Conv1d)]
    if len(convs)!=2 or any(m.kernel_size!=(3,) or m.stride!=(1,) or m.padding!=(1,) or m.dilation!=(1,) for m in convs):
        raise ValueError('Expected two padded, stride-one Conv1d layers with kernel size 3')
    if mode=='bypass':
        conditioner.use_temporal_convolution=False
    elif mode=='center_only':
        with torch.no_grad():
            for conv in convs:
                conv.weight[:,:,0].zero_()
                conv.weight[:,:,2].zero_()
    conditioner._applied_temporal_intervention=mode
    return dict(mode=mode,temporal_enabled=conditioner.use_temporal_convolution,
                convolution_layers=len(convs),kernel_sizes=[list(m.kernel_size) for m in convs],
                residual_connection='unchanged for normal/center_only; entire block bypassed for bypass',
                checkpoint_modified_on_disk=False)
