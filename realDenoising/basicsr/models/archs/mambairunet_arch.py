# The Code Implementation of MambaIR model for Real Image Denoising task
import torch
from timm.models.layers import to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import  to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
import torchinfo
NEG_INF = -1000000


class ChannelAttention(torch.nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat:int, squeeze_factor:int=16):
        super(ChannelAttention, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=num_feat,out_channels= num_feat // squeeze_factor, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=num_feat // squeeze_factor, out_channels=num_feat,kernel_size= 1, padding=0),
            torch.nn.Sigmoid())

    def forward(self, x:torch.Tensor)->torch.Tensor:
        y = self.attention(x)
        return x * y


class CAB(torch.nn.Module):
    def __init__(self, num_feat:int, compress_ratio:int=3, squeeze_factor:int=30):
        super(CAB, self).__init__()

        self.cab = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_feat, out_channels=num_feat // compress_ratio, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=num_feat // compress_ratio, out_channels=num_feat, kernel_size=3, stride=1, padding=1),
            ChannelAttention(num_feat=num_feat,squeeze_factor=squeeze_factor)
            )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.cab(x)


##########################################################################
class Mlp(torch.nn.Module):
    def __init__(self, in_features:int, hidden_features:int=None, out_features:int=None, act_layer:torch.nn.Module=torch.nn.GELU, drop:float=0.0):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features=in_features,out_features= hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(in_features=hidden_features,out_features=out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(torch.nn.Module):
    def __init__(self, dim:int, num_heads:int):
        super(DynamicPosBias,self).__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = torch.nn.Linear(in_features=2, out_features=self.pos_dim)
        self.pos1 = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim, out_features=self.pos_dim),
        )
        self.pos2 = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim,out_features= self.pos_dim)
        )
        self.pos3 = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim,out_features= self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


#########################################
class Attention(torch.nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim:int, num_heads:int, qkv_bias:bool=True, qk_scale:float=None, attn_drop:float=0.0, proj_drop:float=0.0,
                 position_bias=True):
        super(Attention,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(dim=self.dim // 4,num_heads= self.num_heads)
        self.qkv = torch.nn.Linear(in_features=dim,out_features= dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(p=attn_drop)
        self.proj = torch.nn.Linear(in_features=dim,out_features= dim)
        self.proj_drop = torch.nn.Dropout(p=proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor, h, w, mask=None)->torch.Tensor:
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (h, w)
        B_, N, C = x.shape
        assert h * w == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(dim0=0, dim1=1).contiguous().float()  # (2h-1)*(2w-1) 2
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw
            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class SS2D(torch.nn.Module):
    def __init__(
            self,
            d_model,
            d_state:int=16,
            d_conv:int=3,
            expand:int=2,
            dt_rank:str="auto",
            dt_min:float=0.001,
            dt_max:float=0.1,
            dt_init="random",
            dt_scale:float=1.0,
            dt_init_floor:float=1e-4,
            dropout:float=0.0,
            conv_bias:bool=True,
            bias:bool=False,
            device:torch.device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SS2D,self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = torch.nn.Linear(in_features=self.d_model, out_features=self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = torch.nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = torch.nn.SiLU()

        self.x_proj = (
            torch.nn.Linear(in_features=self.d_inner,out_features= (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner, out_features=(self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner,out_features= (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner, out_features=(self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = torch.nn.Parameter(data=torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = torch.nn.Parameter(data=torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = torch.nn.Parameter(data=torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = torch.nn.LayerNorm(normalized_shape=self.d_inner)
        self.out_proj = torch.nn.Linear(in_features=self.d_inner,out_features= self.d_model, bias=bias, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = torch.nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            torch.nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
           tensor= torch.arange(start=1, end=d_state + 1, dtype=torch.float32, device=device),
            pattern="n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(input=A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(tensor=A_log, pattern="d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = torch.nn.Parameter(data=A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(tensor=D, pattern="n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = torch.nn.Parameter(data=D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, c, h, w = x.shape
        L = h * w
        K = 4

        x_hwwh = torch.stack(tensors=[x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat(tensors=[x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(tensor=x_dbl,split_size_or_sections= [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, w, h), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, w, h), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs)->torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * torch.nn.functional.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.0,
            norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0.0,
            d_state: int = 16,
            expand: float = 2.0,
            **kwargs,
    ):
        super(VSSBlock,self).__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=int(expand),dropout=attn_drop_rate, **kwargs)
        self.drop_path = torch.nn.Dropout(drop_path)
        self.skip_scale= torch.nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = torch.nn.LayerNorm(hidden_dim)
        self.skip_scale2 = torch.nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input:torch.Tensor, x_size)->torch.Tensor:
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(torch.nn.Module):
    def __init__(self, in_c:int=3, embed_dim:int=48, bias:bool=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = torch.nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.proj(x)
        x = rearrange(tensor=x, pattern="b c h w -> b (h w) c").contiguous()
        return x


##########################################################################
## Resizing modules
class Downsample(torch.nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_feat,out_channels= n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  torch.nn.PixelUnshuffle(2))

    def forward(self, x:torch.Tensor, H, W)->torch.Tensor:
        x = rearrange(tensor=x, pattern="b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(tensor=x, pattern="b c h w -> b (h w) c").contiguous()
        return x


class Upsample(torch.nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  torch.nn.PixelShuffle(upscale_factor=2))

    def forward(self, x:torch.Tensor, h, w)->torch.Tensor:
        x = rearrange(tensor=x, pattern="b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = self.body(x)
        x = rearrange(tensor=x, pattern="b c h w -> b (h w) c").contiguous()
        return x



class MambaIRUNet(torch.nn.Module):
    def __init__(self,
                 inp_channels:int=3,
                 out_channels:int=3,
                 dim:int=48,
                 num_blocks=[4, 6, 6, 8],
                 mlp_ratio:float=2.0,
                 num_refinement_blocks:int=4,
                 drop_path_rate:float=0.0,
                 bias:bool=False,
                 dual_pixel_task:bool=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(MambaIRUNet, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels,embed_dim= dim)
        base_d_state = 4
        self.encoder_level1 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state* 2 ** 2,
            )
            for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(n_feat=dim)  ## From Level 1 to Level 2
        self.encoder_level2 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(n_feat=int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(n_feat= int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state / 2 * 2 ** 3),
            )
            for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(n_feat=int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = torch.nn.Conv2d(in_channels=int(dim * 2 ** 3),out_channels= int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(n_feat=int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = torch.nn.Conv2d(in_channels=int(dim * 2 ** 2),out_channels= int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(n_feat=int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_blocks[0])])

        self.refinement = torch.nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for _ in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = torch.nn.Conv2d(in_channels=dim, out_channels=int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = torch.nn.Conv2d(in_channels=int(dim * 2 ** 1), out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img:torch.Tensor)->torch.Tensor:
        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat(tensors=[inp_dec_level3, out_enc_level3], dim=2)
        inp_dec_level3 = rearrange(tensor=inp_dec_level3,pattern= "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(tensor=inp_dec_level3, pattern="b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat(tensors=[inp_dec_level2, out_enc_level2], dim=2)
        inp_dec_level2 = rearrange(tensor=inp_dec_level2, pattern="b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(tensor=inp_dec_level2, pattern="b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat(tensors=[inp_dec_level1, out_enc_level1], dim=2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(tensor=out_dec_level1, pattern="b (h w) c -> b c h w", h=H, w=W).contiguous()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    height = 128
    width = 128
    model = MambaIRUNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False
    )
    model=model.to(device="cuda")
    # print(model)
    x = torch.randn((1, 3, height, width)).to(device="cuda")
    torchinfo.summary(model=model,input_data=x)