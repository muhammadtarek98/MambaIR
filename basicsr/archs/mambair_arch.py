# Code Implementation of the MambaIR Model
import math
import torch
from timm.models.layers import  to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import  repeat

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
            torch.nn.Conv2d(in_channels=num_feat,
                            out_channels= num_feat // squeeze_factor,
                            kernel_size=1,
                            padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=num_feat // squeeze_factor,
                            out_channels= num_feat,
                            kernel_size= 1,
                            padding=0),
            torch.nn.Sigmoid())

    def forward(self, x:torch.Tensor)->torch.Tensor:
        y = self.attention(x)
        return x * y


class CAB(torch.nn.Module):
    def __init__(self, num_feat:int, is_light_sr:bool= False,
                 compress_ratio:int=3,squeeze_factor:int=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_feat,
                            out_channels= num_feat // compress_ratio,
                            kernel_size= 3,
                            stride= 1,
                            padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=num_feat // compress_ratio,
                            out_channels= num_feat,
                            kernel_size= 3,
                            stride= 1,
                            padding=1),
            ChannelAttention(num_feat=num_feat,squeeze_factor= squeeze_factor)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.cab(x)


class Mlp(torch.nn.Module):
    def __init__(self, in_features:int,
                 hidden_features:int=None,
                 out_features:int=None,
                 act_layer=torch.nn.GELU, drop:float=0.0):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                   out_features= hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(in_features=hidden_features,out_features= out_features)
        self.drop = torch.nn.Dropout(p=drop)

    def forward(self, x:torch.Tensor)->torch.Tensor:
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
        self.pos_proj = torch.nn.Linear(in_features=2,out_features= self.pos_dim)
        self.pos1 = torch.nn.Sequential(
            torch.nn.LayerNorm(self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim,out_features= self.pos_dim),
        )
        self.pos2 = torch.nn.Sequential(
            torch.nn.LayerNorm(self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim,out_features= self.pos_dim)
        )
        self.pos3 = torch.nn.Sequential(
            torch.nn.LayerNorm(self.pos_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=self.pos_dim,out_features= self.num_heads)
        )

    def forward(self, biases:torch.Tensor)->torch.Tensor:
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


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
    def __init__(self, dim:int, num_heads:int,
                 qkv_bias:bool=True, qk_scale:int=None,
                 attn_drop:float=0.0, proj_drop:float=0.0,
                 position_bias:bool=True):
        super(Attention,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)
        self.qkv = torch.nn.Linear(in_features=dim,out_features= dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(in_features=dim,out_features= dim)
        self.proj_drop = torch.nn.Dropout(p=proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x:torch.Tensor, H, W, mask=None)->torch.Tensor:
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W
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
            expand:float=2.0,
            dt_rank:str="auto",
            dt_min:float=0.001,
            dt_max:float=0.1,
            dt_init:str="random",
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
        self.in_proj = torch.nn.Linear(in_features=self.d_model,out_features= self.d_inner * 2, bias=bias, **factory_kwargs)
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
            torch.nn.Linear(in_features=self.d_inner,
                            out_features= (self.dt_rank + self.d_state * 2),
                            bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner,
                            out_features=(self.dt_rank + self.d_state * 2),
                            bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner,
                            out_features= (self.dt_rank + self.d_state * 2),
                            bias=False, **factory_kwargs),
            torch.nn.Linear(in_features=self.d_inner,
                            out_features= (self.dt_rank + self.d_state * 2),
                            bias=False, **factory_kwargs),
        )
        self.x_proj_weight = torch.nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
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
        self.dt_projs_weight = torch.nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = torch.nn.Parameter(torch.stack((t.bias for t in self.dt_projs), dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.out_norm = torch.nn.LayerNorm(self.d_inner)
        self.out_proj = torch.nn.Linear(in_features=self.d_inner,
                                        out_features= self.d_model,
                                        bias=bias, **factory_kwargs)
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank:int, d_inner:int,
                dt_scale:float=1.0, dt_init:str="random",
                dt_min:float=0.001,dt_max:float=0.1,
                dt_init_floor:float=1e-4,**factory_kwargs):
        dt_proj = torch.nn.Linear(in_features=dt_rank,
                                  out_features= d_inner,
                                  bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            torch.nn.init.constant_(dt_proj.weight, dt_init_std)
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
    def A_log_init(d_state:int, d_inner, copies:int=1, device:torch.device=None, merge:bool=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = torch.nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies:int=1, device:torch.device=None, merge:bool=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = torch.nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(tensors=[x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat(tensors=[x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
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
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
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
            norm_layer= torch.nn.LayerNorm,
            attn_drop_rate: float = 0.0,
            d_state: int = 16,
            expand: float = 2.0,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super(VSSBlock,self).__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = torch.nn.Dropout(p=drop_path)
        self.skip_scale= torch.nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = torch.nn.LayerNorm(hidden_dim)
        self.skip_scale2 = torch.nn.Parameter(torch.ones(hidden_dim))
    def forward(self, input:torch.Tensor, x_size:list)->torch.Tensor:
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(torch.nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 dim:int,
                 input_resolution:tuple[int]|list[int],
                 depth:int,
                 drop_path:float=0.0,
                 d_state:int=16,
                 mlp_ratio:float=2.0,
                 norm_layer:torch.nn.Module=torch.nn.LayerNorm,
                 downsample=None,
                 use_checkpoint:bool=False,
                 is_light_sr:bool=False):
        super(BasicLayer).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=torch.nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x:torch.Tensor, x_size)->torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class MambaIR(torch.nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.
       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The  reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size:int=64,
                 patch_size:int=1,
                 in_chans:int=3,
                 embed_dim:int=96,
                 depths:tuple[int]=(6, 6, 6, 6),
                 drop_rate:float=0.0,
                 d_state:int = 16,
                 mlp_ratio:float=2.0,
                 drop_path_rate:float=0.1,
                 norm_layer:torch.nn.Module=torch.nn.LayerNorm,
                 patch_norm:bool=True,
                 use_checkpoint:bool=False,
                 upscale:int=2,
                 img_range:float=1.0,
                 upsampler:str='',
                 resi_connection:str='1conv',
                 **kwargs):
        """    upscale: 4
    in_chans: 3
    img_size: 64
    window_size: 16
    compress_ratio: 3
    squeeze_factor: 30
    conv_scale: 0.01
    overlap_ratio: 0.5
    img_range: 1.0
    depths: [6, 6, 6, 6, 6, 6]
    embed_dim: 180
    num_heads: [6, 6, 6, 6, 6, 6]
    mlp_ratio: 2
    upsampler: pixelshuffle
    resi_connection: 1conv"""
        super(MambaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = torch.nn.Conv2d(in_channels=num_in_ch,
                                          out_channels= embed_dim,
                                          kernel_size= 3,
                                          stride=1,
                                          padding=1)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = torch.nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]  # stochastic depth decay rule
        # build Residual State Space Group (RSSG)
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = torch.nn.Conv2d(in_channels=embed_dim,
                                                   out_channels=embed_dim,
                                                   kernel_size= 3,
                                                   stride=1,
                                                   padding=1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 4,kernel_size= 3, stride=1,padding= 1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv2d(in_channels=embed_dim // 4,out_channels= embed_dim // 4,kernel_size= 1, stride=1,padding= 0),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv2d(in_channels=embed_dim // 4,out_channels= embed_dim,kernel_size= 3,stride= 1,padding= 1))
        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embed_dim,out_channels= num_feat,kernel_size= 3, stride=1,padding= 1),
                torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = torch.nn.Conv2d(in_channels=num_feat,out_channels= num_out_ch,kernel_size= 3,stride= 1, padding=1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            # for image denoising
            self.conv_last = torch.nn.Conv2d(in_channels=embed_dim,out_channels= num_out_ch, kernel_size=3, stride=1,padding= 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch. nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        return x
    def forward(self, x:torch.Tensor)->torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
        x = x / self.img_range + self.mean
        return x
    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ResidualGroup(torch.nn.Module):
    """Residual State Space Group (RSSG).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """
    def __init__(self,
                 dim:int,
                 input_resolution:list,
                 depth:int,
                 d_state:int=16,
                 mlp_ratio:float=4.0,
                 drop_path=0.0,
                 norm_layer:torch.nn.Module=torch.nn.LayerNorm,
                 downsample=None,
                 use_checkpoint:bool=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection:str='1conv',
                 is_light_sr:bool = False):
        super(ResidualGroup, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)
        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = torch.nn.Conv2d(in_channels=dim,out_channels= dim,kernel_size= 3,stride= 1, padding=1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=dim,out_channels= dim // 4,kernel_size= 3, stride=1,padding= 1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv2d(in_channels=dim // 4,out_channels= dim // 4,kernel_size= 1,stride= 1,padding= 0),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv2d(in_channels=dim // 4,out_channels= dim,kernel_size= 3, stride=1, padding=1))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x:torch.Tensor, x_size:list)->torch.Tensor:
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed(torch.nn.Module):
    r""" transfer 2D feature map into 1D token sequence
    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size:int=224,
                 patch_size:int=4, in_chans:int=3,
                 embed_dim:int=96, norm_layer:torch.nn.Module=None):
        super(PatchEmbed,self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.flatten(2).transpose(dim0=1,dim1= 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(torch.nn.Module):
    r""" return 2D feature map from 1D token sequence
    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size:int=224,
                 patch_size:int=4, in_chans:int=3,
                 embed_dim:int=96,norm_layer:torch.nn.Module=None):
        super(PatchUnEmbed,self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x:torch.Tensor, x_size:list)->torch.Tensor:
        x = x.transpose(dim0=1,dim1= 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(torch.nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale:int, num_feat:int, num_out_ch:int):
        self.num_feat = num_feat
        m = []
        m.append(torch.nn.Conv2d(in_channels=num_feat,out_channels= (scale**2) * num_out_ch, kernel_size=3,stride= 1, padding=1))
        m.append(torch.nn.PixelShuffle(upscale_factor=scale))
        super(UpsampleOneStep, self).__init__(*m)

class Upsample(torch.nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale:int, num_feat:int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(torch.nn.Conv2d(in_channels=num_feat,out_channels= 4 * num_feat, kernel_size=3, stride=1, padding=1))
                m.append(torch.nn.PixelShuffle(upscale_factor=2))
        elif scale == 3:
            m.append(torch.nn.Conv2d(in_channels=num_feat,out_channels= 9 * num_feat,kernel_size= 3, stride=1,padding= 1))
            m.append(torch.nn.PixelShuffle(upscale_factor=3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)