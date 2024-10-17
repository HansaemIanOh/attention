import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float=0.,
        attn_dropout: float=0,
        act: nn.Module=nn.GELU(),
        **kwargs
    ) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.act = act
        self.q_proj = nn.Linear(attn_dim, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        # LN -> MHA -> RES -> LN -> MLP -> RES
        self.LN1 = nn.LayerNorm(attn_dim)
        self.LN2 = nn.LayerNorm(attn_dim)
        self.MLP = nn.Sequential(
            nn.Linear(attn_dim, mlp_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, attn_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: Tensor, dis_attn_mask: Tensor, cls_attn_mask: Tensor) -> Tensor:
        '''
        x : [B, S, F]
        pad_attn_mask : [B, S]
        '''
        h = x
        res = x
        h = self.LN1(h) # LN
        
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        B, S, F = k.shape
        q = q.view(B, self.num_heads, S, self.head_dim)
        k = k.view(B, self.num_heads, S, self.head_dim)
        v = v.view(B, self.num_heads, S, self.head_dim)
        # q : BHSF | k, v : BHsF, S = s
        w = torch.einsum('BHSF, BHFs -> BHSs', q, k.transpose(-2, -1)) * (self.head_dim ** (-0.5))
        # Apply attention masks
        w = w + dis_attn_mask.unsqueeze(1) + cls_attn_mask.unsqueeze(1)
        w = nn.functional.softmax(w, dim=-1) # dimension of s
        h_attn = torch.einsum('BHSs, BHsF -> BHSF', w, v)
        h_attn = h_attn.view(B, S, F)
        h = h_attn + res # RES
        h = self.LN2(h) # LN
        res = h
        h = self.MLP(h) # MLP
        h = h + res # RES
        return h
