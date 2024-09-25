import math
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from MambaIR.realDenoising.basicsr.utils import get_root_logger

@torch.no_grad()
def default_init_weights(module_list:list, scale:float=1.0, bias_fill:float=0.0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                torch.nn.init.constant_(m.weight, val=1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block:torch.nn.Module, num_basic_block:int, **kwarg)->torch.nn.Sequential:
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return torch.nn.Sequential(*layers)


class ResidualBlockNoBN(torch.nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat:int=64, res_scale:float=1, pytorch_init:bool=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = torch.nn.Conv2d(in_channels=num_feat,out_channels= num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights(module_list=[self.conv1, self.conv2], scale=0.1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(torch.nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale:int, num_feat:int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, base=2))):
                m.append(torch.nn.Conv2d(in_channels=num_feat,out_channels= 4 * num_feat,kernel_size= 3, stride=1, padding=1))
                m.append(torch.nn.PixelShuffle(upscale_factor=2))
        elif scale == 3:
            m.append(torch.nn.Conv2d(in_channels=num_feat, out_channels=9 * num_feat, kernel_size=3, stride=1, padding=1))
            m.append(torch.nn.PixelShuffle(upscale_factor=3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x:torch.Tensor,
              flow:torch.Tensor,
              interp_mode:str='bilinear',
              padding_mode:str='zeros',
              align_corners:bool=True)->torch.Tensor:
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack(tensors=(grid_x, grid_y), dim=2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack(tensors=(vgrid_x, vgrid_y), dim=3)
    output = torch.nn.functional.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow:torch.Tensor,
                size_type:str,
                sizes:list,
                interp_mode:str='bilinear',
                align_corners:bool=False)->torch.Tensor:
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = torch.nn.functional.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x:torch.Tensor, scale:int)->torch.Tensor:
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)
