 # coding=utf-8
#from inspect import classify_class_attrs
#from select import select
#from turtle import forward
#from typing_extensions import Self
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
from models.swinNet import SwinTransformer
import time
import math
from functools import partial
from typing import Optional, Callable
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
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
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
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

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
class SE(nn.Module):
    def __init__(self):
        super(SE,self).__init__()
        self.conv_1x1_0 = nn.Conv2d(128,32,kernel_size=1)
        self.conv_1x1_1 = nn.Conv2d(128,32,kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(1024,32,kernel_size=1)
        self.CA_4_1 = ChannelAttention(32*2)
        self.CA_1_0 = ChannelAttention(32*2)
        self.conv_3x3_1 = nn.Conv2d(32*2,32,kernel_size=3,padding=1)
        self.conv_3x3_0 = nn.Conv2d(32*2,32,kernel_size=3,padding=1)
        self.conv_1x1_out = nn.Conv2d(32,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,x0,x1,x4):
        f0 = self.conv_1x1_0(x0)
        f1 = self.conv_1x1_1(x1)
        f4 = self.conv_1x1_4(x4)
        S1 = self.conv_3x3_1(self.CA_4_1(torch.cat((f1,self.upsample3(f4)),1)))
        S0 = self.conv_3x3_0(self.CA_1_0(torch.cat((f0,S1),1)))
        out = F.sigmoid(self.conv_1x1_out(S0))
        return out


class FF(nn.Module):
    def __init__(self):
        super(FF,self).__init__()
        self.conv1 = nn.Conv2d(32*2,32,kernel_size=3,padding=1)
        self.CA = ChannelAttention_1(32)
        #self.conv2 = nn.Conv2d(64,1,kernel_size=1)
    def forward(self,x1,x2):
        c = self.conv1(torch.cat((x1,x2),1))
        v = self.CA(c)
        #y1 = x1*v
        #y2 = x2*v
        out = c*v+c
        return out
class FFF(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FFF,self).__init__()
        self.conv_1 = nn.Conv2d(2*in_channels,out_channels,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(2*in_channels,out_channels,kernel_size=3,padding=1)
        self.CA = ChannelAttention_1(out_channels)
        self.SA = SpatialAttention()
    def forward(self,x1,x2,x3):
        c1 = self.conv_1(torch.cat((x1,x2),1))
        c2 = self.conv_2(torch.cat((x2,x3),1))
        v = self.CA(c1)
        w = self.SA(c2)
        out = c1*v+c2*w+c1+c2
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


#mid level
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x





class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class RTI(nn.Module):
    def __init__(self,in_channels):
        super(RTI,self).__init__()
        self.ca_r = ChannelAttention_1(in_channels)
        self.ca_t = ChannelAttention_1(in_channels)
        self.sa_r = SpatialAttention()
        self.sa_t = SpatialAttention()

        self.ca_rt = ChannelAttention_1(in_channels)
        self.sa_rt = SpatialAttention()
        #self.conv_1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv_1 = CBR(3*in_channels,in_channels)
    def forward(self,r,t):
        #r = r.permute(0, 3, 1,2).contiguous()
        #t = t.permute(0, 3, 1,2).contiguous()
        r_ca = self.ca_r(r)*r+r
        t_ca = self.ca_t(t)*t+t

        s_r = self.sa_r(r_ca)
        s_t = self.sa_t(t_ca)

        max_s, _ = torch.max(torch.cat((s_r,s_t),1), dim=1, keepdim=True)

        r_sa = r_ca*max_s+r_ca
        t_sa = t_ca*max_s+t_ca

        rt = r*t+r+t
        rt_ca = self.ca_rt(rt)*rt+rt
        rt_sa = self.sa_rt(rt_ca)*rt_ca+rt_ca

        out = self.conv_1(torch.cat((r_sa,t_sa,rt_sa),1))
        

        return out
class PFM(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(PFM,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )
        
        self.ca0 = ChannelAttention_1(in_channels)
        self.ca1 = ChannelAttention_1(in_channels)
        self.ca2 = ChannelAttention_1(in_channels)
        self.ca3 = ChannelAttention_1(in_channels)
        self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y0 = self.ca0(y0)+y0
        y1 = self.branch1(x+y0)
        y1 = self.ca0(y1)+y1
        y2 = self.branch2(x+y1)
        y2 = self.ca0(y2)+y2
        y3 = self.branch3(x+y2)
        y3 = self.ca0(y3)+y3

        y = self.conv(torch.cat((y0,y1,y2,y3),1))+x
        return y
class PFM_1(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(PFM_1,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )
        
        self.ca0 = ChannelAttention_1(in_channels)
        self.ca1 = ChannelAttention_1(in_channels)
        self.ca2 = ChannelAttention_1(in_channels)
        self.ca3 = ChannelAttention_1(in_channels)
        self.conv = nn.Conv2d(3*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y0 = self.ca0(y0)+y0
        y1 = self.branch1(x+y0)
        y1 = self.ca0(y1)+y1
        y2 = self.branch2(x+y1)
        y2 = self.ca0(y2)+y2
        

        y = self.conv(torch.cat((y0,y1,y2),1))+x
        return y
    
class PFM_2(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(PFM_2,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )
        
        self.ca0 = ChannelAttention_1(in_channels)
        self.ca1 = ChannelAttention_1(in_channels)
        self.ca2 = ChannelAttention_1(in_channels)
        self.ca3 = ChannelAttention_1(in_channels)
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y0 = self.ca0(y0)+y0
        y1 = self.branch1(x+y0)
        y1 = self.ca0(y1)+y1
        
        

        y = self.conv(torch.cat((y0,y1),1))+x
        return y
class GCANet(nn.Module):#输入三通道
    def __init__(self):
        super(GCANet, self).__init__()
        self.rgb_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        
        

        self.conv_3x3_0_RGB = nn.Conv2d(128*2,32,1)
        self.conv_3x3_1_RGB = nn.Conv2d(128*2,32,1)
        self.conv_3x3_2_RGB = nn.Conv2d(256*2,32,1)
        self.conv_3x3_3_RGB = nn.Conv2d(512*2,32,1)
        self.conv_3x3_4_RGB = nn.Conv2d(1024*2,32,1)

        self.conv_3x3_0_T = nn.Conv2d(128,64,1)
        self.conv_3x3_1_T = nn.Conv2d(128,64,1)
        self.conv_3x3_2_T = nn.Conv2d(256,64,1)
        self.conv_3x3_3_T = nn.Conv2d(512,64,1)
        self.conv_3x3_4_T = nn.Conv2d(1024,64,1)

        self.conv_3x3_0_de = nn.Conv2d(64*2,64,1)
        self.conv_3x3_1_de= nn.Conv2d(64*2,64,1)
        self.conv_3x3_2_de = nn.Conv2d(64*2,64,1)
        self.conv_3x3_3_de = nn.Conv2d(64*2,64,1)

        
        

        self.pm0 = PFM(64)
        self.pm1 = PFM_1(64)
        self.pm2 = PFM_1(64)
        self.pm3 = PFM_2(64)
        self.pm4 = PFM_2(64)
        
        
        self.rti0 =RTI(128)
        self.rti1 =RTI(128)
        self.rti2 =RTI(256)
        self.rti3 =RTI(512)
        self.rti4 =RTI(1024)
        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv_out_4 = nn.Conv2d(64,1,kernel_size=1)
        self.conv_out_3 = nn.Conv2d(64,1,kernel_size=1)
        self.conv_out_2 = nn.Conv2d(64,1,kernel_size=1)
        self.conv_out_1 = nn.Conv2d(64,1,kernel_size=1)
        self.conv_out_0 = nn.Conv2d(64,1,kernel_size=1)
        self.up_2 = PatchExpand2D(dim=64)
        dims_decoder=[64,64,64,64,64]
        depths_decoder = [2,2,2,2,2]
        self.layers_up = nn.ModuleList()
        dpr_decoder=[x.item() for x in torch.linspace(0, 0.2, sum(depths_decoder))]
        for i_layer in range(5):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if 16 is None else 16, # 20240109
                drop=0.3, 
                attn_drop=0,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                #upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=False,
            )
            self.layers_up.append(layer)
            
        depths_en=[2,2,2,2,2]
        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(depths_en))]
        self.layers_en = nn.ModuleList()
        dims_en = [64,64,64,64,64]
        
        for i_layer in range(5):
            layer = VSSLayer(
                dim=dims_en[i_layer],
                depth=depths_en[i_layer],
                d_state=math.ceil(dims_en[0] / 6) if 16 is None else 16, # 20240109
                drop=0.3, 
                attn_drop=0,
                drop_path=dpr[sum(depths_en[:i_layer]):sum(depths_en[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                #downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
            )
            self.layers_en.append(layer)
        
    
    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
      
        
    def forward(self, x_rgb,x_t):
        
        r0,r1,r2,r3,r4 = self.rgb_swin(x_rgb)
        t0,t1,t2,t3,t4 = self.rgb_swin(x_t)

        
        
        

        T0 = self.conv_3x3_0_T(self.rti0(r0,t0))
        T1 = self.conv_3x3_1_T(self.rti1(r1,t1))
        T2 = self.conv_3x3_2_T(self.rti2(r2,t2))
        T3 = self.conv_3x3_3_T(self.rti3(r3,t3))
        T4 = self.conv_3x3_4_T(self.rti4(r4,t4))

        T0 = T0.permute(0, 2, 3,1).contiguous()
        T0 = self.layers_en[0](T0)
        T0 = T0.permute(0, 3, 1,2).contiguous()

        T1 = T1.permute(0, 2, 3,1).contiguous()
        T1 = self.layers_en[1](T1)
        T1 = T1.permute(0, 3, 1,2).contiguous()

        T2 = T2.permute(0, 2, 3,1).contiguous()
        T2 = self.layers_en[2](T2)
        T2 = T2.permute(0, 3, 1,2).contiguous()

        T3 = T3.permute(0, 2, 3,1).contiguous()
        T3 = self.layers_en[3](T3)
        T3 = T3.permute(0, 3, 1,2).contiguous()

        T4 = T4.permute(0, 2, 3,1).contiguous()
        T4 = self.layers_en[4](T4)
        T4 = T4.permute(0, 3, 1,2).contiguous()

        P4 = self.pm4(T4)
        P3 = self.pm3(T3)
        P2 = self.pm2(T2)
        P1 = self.pm1(T1)
        P0 = self.pm0(T0)

        P4 = P4.permute(0, 2, 3,1).contiguous()

        F4 = self.layers_up[0](P4)
        F4 = self.upsample1(F4.permute(0, 3, 1,2).contiguous())
        F3 = self.conv_3x3_3_de(torch.cat((F4,P3),1))
        F3 = F3.permute(0, 2, 3,1).contiguous()
        F3 = self.layers_up[1](F3)
        
        F3 = self.upsample1(F3.permute(0, 3, 1,2).contiguous())
        F2 = self.conv_3x3_2_de(torch.cat((F3,P2),1))
        F2 = F2.permute(0, 2, 3,1).contiguous()
        
        F2 = self.layers_up[2](F2)
        F2 = self.upsample1(F2.permute(0, 3, 1,2).contiguous())
        F1 = self.conv_3x3_1_de(torch.cat((F2,P1),1))
        F1 = F1.permute(0, 2, 3,1).contiguous()
        F1 = self.layers_up[3](F1)
        F1 = F1.permute(0, 3, 1,2).contiguous()
        F0 = self.conv_3x3_0_de(torch.cat((F1,P0),1))
        F0 = F0.permute(0, 2, 3,1).contiguous()
        
        F0 = self.layers_up[4](F0)
        f0 = F0.permute(0, 3, 1,2).contiguous()
        f1 = F1
        f2 = F2
        f3 = F3
        f4 = F4


        

        

        Sal0 = self.conv_out_0(f0)
        Sal1 = self.conv_out_1(f1)
        Sal2 = self.conv_out_2(f2)
        Sal3 = self.conv_out_3(f3)
        Sal4 = self.conv_out_4(f4)
        return self.upsample2(Sal0),self.upsample2(Sal1),self.upsample2(Sal2),self.upsample3(Sal3),self.upsample4(Sal4)


