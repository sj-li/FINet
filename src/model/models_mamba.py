import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import DropPath
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

PACKET_NUM = 5  # 5 packets constitutes an flow array

from .layers.mamba.vim_mamba import create_block

class Mamba_Layer(nn.Module):
    def __init__(self, embed_dim, num_layers, drop_path=0., dtype=None): # , bimamba_type="none"
        super().__init__()
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        for i in range(num_layers):
            self.layers = nn.ModuleList([
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    # if_bimamba=False,
                # bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                # if_devide_out=True,
                # init_layer_scale=None,
            )  for i in range(num_layers)])

            # 依照第148 行，为layers添加一层norm
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, hidden_states, inference_params=None):
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, inference_params=inference_params)
        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )        
        return hidden_states


class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, embed_dim, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.mamba(x)
        # x = self.norm2(x)

        return x
    

class NATM_Layer(nn.Module):
    def __init__(self, nat_layer, mamba_layer, embed_dim, dropout):
        super(NATM_Layer, self).__init__()
        self.nat_layer = nat_layer
        self.mamba_layer = mamba_layer
    def forward(self, x):
        x = self.nat_layer(x)
        x = self.mamba_layer(x)

        return x
    
class NerATM_Layer(nn.Module):
    def __init__(self, natm_layer, mamba_layer, embed_dim, dropout):
        super(NerATM_Layer, self).__init__()
        self.natm_layer = natm_layer
        self.mamba_layer = mamba_layer
        self.norm_attn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        shortcut = x
        x = self.natm_layer(x)
        x = shortcut + self.norm_attn(x)
        x = self.mamba_layer(x)

        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)            # [56,50,embed_dim] -> [56,50,embed_dim*3]
        x = self.act(x)            # nn.GELU()
        x = self.drop(x)           # [56,50,embed_dim*3]
        x = self.fc2(x)            # [56,50,embed_dim*3] -> [56,50,embed_dim]
        x = self.drop(x)
        return x
    

class Samba_Layer(nn.Module):     # for AgentEmbedding module
    def __init__(self, 
                 embed_dim, 
                 num_layers, 
                 drop_path=0., 
                 mlp_ratio=4.0, 
                 drop=0.0,
                #  bimamba_type="none",
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(embed_dim)
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        for i in range(num_layers):
            self.layers = nn.ModuleList([
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    # if_bimamba=False,
                # bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                # if_devide_out=True,
                # init_layer_scale=None,
            )  for i in range(num_layers)])

            # 依照第148 行，为layers添加一层norm
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)

            # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.norm2 = norm_layer(embed_dim)
            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )


    def forward(self, hidden_states, inference_params=None):
        shortcut = hidden_states
        hidden_states = self.norm1(hidden_states)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, inference_params=inference_params)
        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )        
        hidden_states = shortcut + self.drop_path(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        # hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))
        return hidden_states
    
class Samba_Layer_Lane(nn.Module):     # for LaneEmbedding module
    def __init__(self, 
                 embed_dim, 
                 num_layers, 
                 drop_path=0., 
                 mlp_ratio=4.0, 
                 drop=0.0,
                 lane_if_bimamba=True,
                #  bimamba_type="none",
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(embed_dim)
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        for i in range(num_layers):
            self.layers = nn.ModuleList([
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    # if_bimamba=lane_if_bimamba,
                    # bimamba_type=bimamba_type,
                    drop_path=drop_path,
                    # if_devide_out=True,
                    # init_layer_scale=None,
                )  for i in range(num_layers)])

            # 依照第148 行，为layers添加一层norm
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)

            # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.norm2 = norm_layer(embed_dim)
            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )


    def forward(self, hidden_states, inference_params=None):
        shortcut = hidden_states
        hidden_states = self.norm1(hidden_states)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, inference_params=inference_params)
        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )        
        hidden_states = shortcut + self.drop_path(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        # hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))
        return hidden_states
    

    def __init__(self, 
                 embed_dim, 
                 num_layers, 
                 drop_path=0., 
                 mlp_ratio=4.0, 
                 drop=0.0,
                #  bimamba_type="none",
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(embed_dim)
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        for i in range(num_layers):
            self.layers = nn.ModuleList([
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    # if_bimamba=False,
                    # bimamba_type=bimamba_type,
                    drop_path=drop_path,
                    # if_devide_out=True,
                    # init_layer_scale=None,
                )  for i in range(num_layers)])

            # 依照第148 行，为layers添加一层norm
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)

            # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.norm2 = norm_layer(embed_dim)
            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )


    def forward(self, hidden_states, inference_params=None):
        shortcut = hidden_states
        hidden_states = self.norm1(hidden_states)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, inference_params=inference_params)
        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )        
        hidden_states = shortcut + self.drop_path(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        # hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))
        return hidden_states