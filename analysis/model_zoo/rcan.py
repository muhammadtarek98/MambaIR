import math
import torch
import torchinfo

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return torch.nn.Conv2d(
        in_channels=in_channels,out_channels= out_channels,kernel_size= kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(torch.nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(torch.nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=torch.nn.ReLU(True)):

        m = [torch.nn.Conv2d(
            in_channels=in_channels,out_channels= out_channels,kernel_size= kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(torch.nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(torch.nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=torch.nn.ReLU(inplace=True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(torch.nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(torch.nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(torch.nn.PixelShuffle(upscale_factor=2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(torch.nn.PixelShuffle(upscale_factor=3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_du = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=channel,out_channels= channel // reduction, kernel_size=1, padding=0, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=channel // reduction,out_channels= channel, kernel_size=1, padding=0, bias=True),
                torch.nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## add SEResBlock
class SEResBlock(torch.nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1):
        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(torch.nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = torch.nn.Sequential(*modules_body)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=channel // reduction,out_channels= channel, kernel_size=1, padding=0, bias=True),
            torch.nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(torch.nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=torch.nn.ReLU(inplace=True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = torch.nn.Sequential(*modules_body)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(torch.nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=torch.nn.ReLU(inplace=True), res_scale=1) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = torch.nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAN(torch.nn.Module):
    def __init__(self, conv=default_conv):
        super(RCAN, self).__init__()
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 2
        act = torch.nn.ReLU(True)
        # RGB mean for DIV2K 1-800
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # RGB mean for DIVFlickr2K 1-3450
        # rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        n_colors = 3
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean =MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = torch.nn.Sequential(*modules_head)
        self.body = torch.nn.Sequential(*modules_body)
        self.tail = torch.nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


model=RCAN()
x=torch.randn((1,3,128,128))
torchinfo.summary(model=model,input_data=x)