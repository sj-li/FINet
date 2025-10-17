from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionBlock
from .layers.time_decoder import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
from .models_mamba import RMSNorm, rms_norm_fn

class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.multimodal_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        
        self.loc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        x = self.multimodal_proj(x)
        loc = self.loc(x).view(-1, 2)
        return loc, x

class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        enc_layer_1: int = 4,
        enc_layer_2: int = 2,
        dec_layer_1: int = 2,
        dec_layer_2: int = 4
    ) -> None:
        super().__init__()

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim) # same as forecast-mae

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.time_decoder = TimeDecoder(dec_layer_1=dec_layer_1, dec_layer_2=dec_layer_2)
        
        num_layers = 4
        encoder_depth = 4
        bimamba_type="none"
        norm_layer = nn.LayerNorm
        self.samba_blocks1 = nn.ModuleList(
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(enc_layer_1)
            ]
        )
        self.norm_f_1 = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path_1 = DropPath(drop_path)
        self.decoder1 = MultimodalDecoder(embed_dim)

        # dpr = [x.item() for x in torch.linspace(0, drop_path, sum(encoder_depth))]
        self.samba_blocks2 = nn.ModuleList(
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(enc_layer_2) #encoder_depth)
            ]
        )
        self.norm_f_2 = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path_2 = DropPath(drop_path)
        # self.fut_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.fut_mlp = nn.Linear(embed_dim, embed_dim)
        
        self.decoder0 = MultimodalDecoder(embed_dim)
        
        self.future_steps = future_steps
        self.tokens = nn.Parameter(torch.randn(1, 6, 128))   

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        # nn.init.constant_(self.fut_mlp.weight, 1)  # Set all weights to 1
        # nn.init.constant_(self.fut_mlp.bias, 0)    # Set all biases to 0

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)


    def spatial_mamba(self, x_encoder, x_centers):
        ep_offset_1, ep_tok_1 = self.decoder0(x_encoder[:,0])
        # ep_offset_1 = ep_offset_1.detach()

        valid_mask = (x_centers.sum(-1) != 0)
        valid_mask[:,0] = True
        
        center = x_centers[:,0] + ep_offset_1
        dists = ((x_centers - center.unsqueeze(1))**2).sum(-1)
        dists[~valid_mask] = 40000
        dists[:,0] = -1
        dists_sort, indexes = dists.sort(dim=1, descending=True)
        

        x_encoder = torch.gather(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)))
        
        fut_tok = x_encoder[:,-1:].clone()
        fut_tok = fut_tok + self.tokens
        fut_tok = torch.cat([fut_tok[:,:1] + ep_tok_1.unsqueeze(1), fut_tok[:,1:]], 1)

        x_encoder = torch.cat([x_encoder, fut_tok], 1)

        # apply Samba blocks ----------------
        #! First round: init sort & predict ego endpoint
        residual = None
        for blk in self.samba_blocks1:
            x_encoder, residual = blk(x_encoder, residual)                                      # [bs, N+M, D]
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_1, RMSNorm) else layer_norm_fn
        x_encoder = fused_add_norm_fn(
            self.drop_path_1(x_encoder),
            self.norm_f_1.weight,
            self.norm_f_1.bias,
            eps=self.norm_f_1.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]

        fut_tok = x_encoder[:, -6:]
        x_encoder = x_encoder[:, :-6]

        x_encoder = torch.scatter(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)), x_encoder)

        x_ego = fut_tok[:,0]
        ep_offset_2, ep_tok_2 = self.decoder1(x_ego)
        # ep_offset_2 = ep_offset_2.detach()

        
        # center_init = y_hat_init[..., -1, :]
        fut_tok = torch.cat([fut_tok[:,:1] + ep_tok_2.unsqueeze(1), fut_tok[:,1:]], 1)

        center = x_centers[:,0] + ep_offset_2
        dists = ((x_centers - center.unsqueeze(1))**2).sum(-1)
        dists[~valid_mask] = 40000
        dists[:,0] = -1
        dists_sort, indexes = dists.sort(dim=1, descending=True)

        x_encoder = torch.gather(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)))
        x_encoder = torch.cat([x_encoder, fut_tok], 1)
        #! Second round: resort & predict ego endpoint
        # for blk in self.samba_blocks2:
        #     x_encoder = blk(x_encoder)                                      # [bs, N+M, D]
        residual = None
        for blk in self.samba_blocks2:
            x_encoder, residual = blk(x_encoder, residual)   
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_2, RMSNorm) else layer_norm_fn
        x_encoder = fused_add_norm_fn(
            self.drop_path_2(x_encoder),
            self.norm_f_2.weight,
            self.norm_f_2.bias,
            eps=self.norm_f_2.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]
        fut_tok = x_encoder[:, -6:]
        x_encoder = x_encoder[:, :-6]
        
        # #! query-base cross-attention
        x_encoder = torch.scatter(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)), x_encoder)
        return x_encoder, fut_tok, [ep_offset_1, ep_offset_2]
        # return x_encoder, fut_tok, [ep_offset_2]



    def forward(self, data):
        

        ###### Scene context encoding ###### 
        # agent encoding
        hist_valid_mask = data["x_valid_mask"] # [16, 48, 50]
        hist_key_valid_mask = data["x_key_valid_mask"] # [16, 48]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        ) # [16, 48, 50, 4] different to forecast-mae

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D) # [768, 50, 4]
        hist_feat_key_valid = hist_key_valid_mask.view(B * N) # [768]
        


        # unidirectional mamba
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous()) # [421, 50, 128]
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual) # [421, 50, 128], [421, 50, 128]
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]

        actor_feat = actor_feat[:, -1] # [421, 128]
        # actor_feat = actor_feat[:, 0] # [421, 128]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        ) # [768, 128]
        actor_feat_tmp[hist_feat_key_valid] = actor_feat # [768, 128]
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1]) # [16, 48, 128]

        # map encoding
        lane_valid_mask = data["lane_valid_mask"] # [16, 125, 20]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2) # [16, 125, 20, 2]
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        ) # [16, 125, 20, 3]
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous()) # [2000, 128]
        lane_feat = lane_feat.view(B, M, -1) # [16, 125, 128]

        # type embedding and position embedding
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat) # [16, 173, 128]

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()] # [16, 48, 128]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()] # [16, 125, 128]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # scene context features
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1) # [16, 173, 128]
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        ) # [16, 173]

        x_encoder = x_encoder + pos_embed # [16, 173, 128]
        
        

        x_encoder, mode, ep_offsets = self.spatial_mamba(x_encoder, x_centers)
        x_encoder = self.norm(x_encoder) # [16, 173, 128]


        x_others = x_encoder[:, 1:N] # [16, 47, 128]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2) # ([16, 47, 60, 2]

        
        ep_embedding = torch.linspace(0, 1, steps=self.future_steps).view(1, 1, -1, 1).to(mode.device) * mode.unsqueeze(2)
        mode = x_encoder[:,:1].unsqueeze(1) + ep_embedding

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, scal, scal_new = \
        self.time_decoder(mode, x_encoder, mask=~key_valid_mask)

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query
            
            "ep_offsets": ep_offsets,

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss
            
        }


        return ret_dict

