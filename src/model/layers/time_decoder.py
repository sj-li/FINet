import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len       # TODO: change dimension
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input) # [16, 6, 60, 128] -> [16, 6, 60, 2]
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001 # [16, 6, 60, 2]
        input = input.max(dim=2)[0]  # ([16, 6, 128] TODO: modify this line first tok / Mamba
        # input = input.mean(dim=2)  # ([16, 6, 128] TODO: modify this line first tok / Mamba
        #input = input[:,:,0,:]
        score = self.score(input).squeeze(-1) # [16, 6]

        return res, score, scal

# class GMMPredictor_dense(nn.Module):
#     def __init__(self, future_len=60, dim=128):
#         super(GMMPredictor_dense, self).__init__()
#         self._future_len = future_len       # TODO: change dimension
#         self.process = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, dim)
#         )
#         self.gaussian = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 2)
#         )
#         self.score = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 1),
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 2)
#         )
    
#     def forward(self, input):
#         input = self.process(input)
#         res = self.gaussian(input) # [16, 6, 60, 128] -> [16, 6, 60, 2]
#         scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001 # [16, 6, 60, 2]
#         input = input.max(dim=2)[0]  # ([16, 6, 128] TODO: modify this line first tok / Mamba
#         #input = input[:,:,0,:]
#         score = self.score(input).squeeze(-1) # [16, 6]

#         return res, score, scal


class TimeDecoder(nn.Module):
    def __init__(self, future_len=60, dim=128, dec_layer_1=4, dec_layer_2=4):
        super(TimeDecoder, self).__init__()

        self.timequery_embed_mamba_1 = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(dec_layer_1)
            ]
        )  
        
        
        self.timequery_norm_f_1 = nn.ModuleList([RMSNorm(dim, eps=1e-5) for i in range(dec_layer_1)])
        self.timequery_drop_path_1 = DropPath(0.2)
        
        self.timequery_embed_mamba_2 = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(dec_layer_2)
            ]
        )  

        self.timequery_norm_f_2 = nn.ModuleList([RMSNorm(dim, eps=1e-5) for i in range(dec_layer_2)])
        self.timequery_drop_path_2 = DropPath(0.2)
        

        # hybrid cross attention
        self.cross_block_dense_1 = nn.ModuleList(
            Cross_Block()
            for i in range(dec_layer_1)
        )
        
        # hybrid cross attention
        self.cross_block_dense_2 = nn.ModuleList(
            Cross_Block()
            for i in range(dec_layer_2)
        )

 
        # MLP for mode query
        self.predictor_1 = GMMPredictor_dense(future_len)

        # MLP for final output
        self.predictor_2 = GMMPredictor_dense(future_len)
        

    def forward(self, mode, encoding, mask=None):
    
        B, M, T, C = mode.shape

        # residual = None
        residuals = None
        for blk_ca, blk_mamba, norm in zip(self.cross_block_dense_1, self.timequery_embed_mamba_1, self.timequery_norm_f_1):
            mode = mode.reshape(B, -1, C) # [16, 360, 128]
            mode = blk_ca(mode, encoding, key_padding_mask=mask)
            mode = mode.reshape(-1, T, C)
            mode, residuals = blk_mamba(mode, residuals) # [16, 60, 128]
            # if residuals is None:
            #     residuals = residual
            # else:
            #     residuals = residuals + residual

        fused_add_norm_fn = rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path_1(mode),
            norm.weight,
            norm.bias,
            eps=norm.eps,
            residual=residuals,
            prenorm=False,
            residual_in_fp32=True  
        ) # [16, 60, 128]
        

        mode = mode.reshape(B, M, T, C)
        y_hat, pi, scal = self.predictor_1(mode) # [16, 6, 60, 2], [16, 6], [16, 6, 60, 2]

        # no residual, no resudual + all residual, lr=0.003 not works
        # residual = None
        residuals = None
        for blk_ca, blk_mamba, norm in zip(self.cross_block_dense_2, self.timequery_embed_mamba_2, self.timequery_norm_f_2):
            mode = mode.reshape(B, -1, C) # [16, 360, 128]
            mode = blk_ca(mode, encoding, key_padding_mask=mask)
            mode = mode.reshape(-1, T, C)
            mode, residuals = blk_mamba(mode, residuals) # [16, 60, 128]
            # if residuals is None:
            #     residuals = residual
            # else:
            #     residuals = residuals + residual
                
        fused_add_norm_fn = rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path_2(mode),
            norm.weight,
            norm.bias,
            eps=norm.eps,
            residual=residuals,
            prenorm=False,
            residual_in_fp32=True  
        ) # [16, 60, 128]


        mode = mode.reshape(B, M, T, C) # [16, 6, 60, 128]

        y_hat_new, pi_new, scal_new = self.predictor_2(mode) # # [16, 6, 60, 2], [16, 6], [16, 6, 60, 2]
        
        dense_pred = None

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, scal, scal_new
    
    
class TimeDecoder_current(nn.Module):
    def __init__(self, future_len=60, dim=128, dec_layer_1=4, dec_layer_2=4):
        super(TimeDecoder_current, self).__init__()

        self.timequery_embed_mamba_1 = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(dec_layer_1)
            ]
        )  
        
        
        self.timequery_norm_f_1 = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path_1 = DropPath(0.2)
        
        self.timequery_embed_mamba_2 = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(dec_layer_2)
            ]
        )  

        self.timequery_norm_f_2 = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path_2 = DropPath(0.2)
        

        # hybrid cross attention
        self.cross_block_dense_1 = nn.ModuleList(
            Cross_Block()
            for i in range(dec_layer_1)
        )
        
        # hybrid cross attention
        self.cross_block_dense_2 = nn.ModuleList(
            Cross_Block()
            for i in range(dec_layer_2)
        )

 
        # MLP for mode query
        self.predictor_1 = GMMPredictor_dense(future_len)

        # MLP for final output
        self.predictor_2 = GMMPredictor_dense(future_len)
        

    def forward(self, mode, encoding, mask=None):
    
        B, M, T, C = mode.shape

        residual = None
        for blk_ca, blk_mamba in zip(self.cross_block_dense_1, self.timequery_embed_mamba_1):
            mode = mode.reshape(B, -1, C) # [16, 360, 128]
            mode = blk_ca(mode, encoding, key_padding_mask=mask)
            mode = mode.reshape(-1, T, C)
            mode, residual = blk_mamba(mode, residual) # [16, 60, 128]

        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f_1, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path_1(mode),
            self.timequery_norm_f_1.weight,
            self.timequery_norm_f_1.bias,
            eps=self.timequery_norm_f_1.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [16, 60, 128]
        

        mode = mode.reshape(B, M, T, C)
        y_hat, pi, scal = self.predictor_1(mode) # [16, 6, 60, 2], [16, 6], [16, 6, 60, 2]


        residual = None
        for blk_ca, blk_mamba in zip(self.cross_block_dense_2, self.timequery_embed_mamba_2):
            mode = mode.reshape(B, -1, C) # [16, 360, 128]
            mode = blk_ca(mode, encoding, key_padding_mask=mask)
            mode = mode.reshape(-1, T, C)
            mode, residual = blk_mamba(mode, residual) # [16, 60, 128]

        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f_2, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path_2(mode),
            self.timequery_norm_f_2.weight,
            self.timequery_norm_f_2.bias,
            eps=self.timequery_norm_f_2.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [16, 60, 128]


        mode = mode.reshape(B, M, T, C) # [16, 6, 60, 128]

        y_hat_new, pi_new, scal_new = self.predictor_2(mode) # # [16, 6, 60, 2], [16, 6], [16, 6, 60, 2]
        
        dense_pred = None

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, scal, scal_new
    

# class GMMPredictor(nn.Module):
#     def __init__(self, future_len=60, dim=128):
#         super(GMMPredictor, self).__init__()
#         self._future_len = future_len
#         self.gaussian = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, self._future_len*2)
#         )
#         self.score = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 1),
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, self._future_len*2)
#         )
    
#     def forward(self, input):
#         B, M, _ = input.shape # [16, 6, 128]
#         res = self.gaussian(input).view(B, M, self._future_len, 2) # [16, 6, 60, 2]
#         scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001 # [16, 6, 120]
#         scal = scal.view(B, M, self._future_len, 2) # ([16, 6, 60, 2]
#         score = self.score(input).squeeze(-1) # [16, 6]

#         return res, score, scal
    
# class GMMPredictor_single(nn.Module):
#     def __init__(self, future_len=60, dim=128):
#         super(GMMPredictor_single, self).__init__()
#         self._future_len = future_len
#         self.gaussian = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, 2)
#         )
#         self.score = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 1),
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, 2)
#         )
    
#     def forward(self, input):
#         B, T, _ = input.shape # [16, 6, 128]
#         res = self.gaussian(input).view(B, -1, self._future_len, 2) # [16, 6, 60, 2]
#         scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001 # [16, 6, 120]
#         scal = scal.view(B, -1, self._future_len, 2) # ([16, 6, 60, 2]
#         score = self.score(input[:,0]).squeeze(-1) # [16, 6]
        
#         res = res.view(-1, 6, self._future_len, 2)
#         scal =scal.view(-1, 6, self._future_len, 2)
#         score = score.view(-1, 6)

#         return res, score, scal
    
# class GMMPredictor_ep(nn.Module):
#     def __init__(self, future_len=60, dim=128):
#         super(GMMPredictor_ep, self).__init__()
#         self._future_len = future_len
#         self.gaussian = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, 2)
#         )
#         self.score = nn.Sequential(
#             nn.Linear(dim, 64), 
#             nn.GELU(), 
#             nn.Linear(64, 1),
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(dim, 256), 
#             nn.GELU(), 
#             nn.Linear(256, 2)
#         )
    
#     def forward(self, input):
#         B, M, _ = input.shape # [16, 6, 128]
#         res = self.gaussian(input).view(B, M, 2) # [16, 6, 60, 2]
#         scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001 # [16, 6, 120]
#         scal = scal.view(B, M, 2) # ([16, 6, 60, 2]
#         score = self.score(input).squeeze(-1) # [16, 6]

#         return res, score, scal
    
