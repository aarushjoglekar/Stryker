import torch
import torch.nn as nn
import copy
import math

# class MultiHeadAttention(nn.Module):
#     """
#     Multi-Head Attention:
#     1) Linear projections for Q, K, V
#     2) Scaled dot-product attention per head
#     3) Concatenate heads and final linear projection
#     https://arxiv.org/pdf/1706.03762
#     """
#     def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
        
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scale = math.sqrt(self.head_dim) # square root of dk for scaling

#         # Separate projections for query, key, and value
#         # HINT: Linear projections for Q, K, V (embed_dim -> embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)

#         # Output projection after concatenating heads (embed_dim -> embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
        
#         # Dropouts
#         self.attn_dropout = nn.Dropout(dropout)
#         self.proj_dropout = nn.Dropout(dropout)

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             query: [batch, seq_q, embed_dim]
#             key:   [batch, seq_k, embed_dim]
#             value: [batch, seq_k, embed_dim]
#         Returns:
#             out:   [batch, seq_q, embed_dim]
#         """
#         B, seq_q, _ = query.size()
#         _, seq_k, _ = key.size()

#         # 1) Project inputs and split into heads
#         q = self.q_proj(query).view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_q, head_dim]
#         k = self.k_proj(key).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)    # [B, heads, seq_k, head_dim]
#         v = self.v_proj(value).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_k, head_dim]

#         # 2) Compute scaled dot-product attention
#         scores = (q @ k.transpose(-2, -1)) / self.scale
#         weights = scores.softmax(dim=-1)
#         weights = self.attn_dropout(weights)
#         attn = weights @ v

#         # 3) Concatenate heads
#         attn = attn.transpose(1, 2).contiguous().view(B, seq_q, self.embed_dim)  # [B, seq_q, embed_dim]

#         # 4) Final projection
#         out = self.out_proj(attn)
#         out = self.proj_dropout(out)

#         return out


# class TransformerEncoderLayer(nn.Module):
#     """
#     Transformer Encoder Layer:
#     1) Multi-head self-attention
#     2) Feed-forward network
#     3) Residual connections + LayerNorm
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         mlp_dim: int,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
        
#         # 1) Self-attention
#         self.self_attn = MultiHeadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#         )

#         # 2) Feed-forward network
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_dim, embed_dim),
#         )

#         # 3) LayerNorm and Dropouts for residuals
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.ff_dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor of shape [batch, seq_len, embed_dim]
#         Returns:
#             Tensor of same shape
#         """
        
#         # 1) Self-attention block
#         attn_out = self.self_attn(x, x, x)
#         x = x + self.attn_dropout(attn_out) # Residual connection
#         x = self.norm1(x)

#         # 2) Feed-forward block
#         ff = self.ffn(x)
#         x = x + self.ff_dropout(ff) # Final drop out and residual connection
#         x = self.norm2(x)

#         return x
    

# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, embed_dim: int):
#         super().__init__()

#         # Clone the encoder_layer num_layers times
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         for encoder_layer in self.layers:
#             # encoder_output = encoder_layer(x)
#             # x = encoder_output
#             x = encoder_layer(x)
            
#         # Apply final normalization
#         x = self.norm(x)
        
#         return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention:
    1) Linear projections for Q, K, V
    2) Scaled dot-product attention per head
    3) Concatenate heads and final linear projection
    https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, embed_dim:int, key_dim:int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim) # square root of dk for scaling

        # Separate projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        
        # Output projection after concatenating heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, embed_dim]
            key:   [batch, seq_k, embed_dim]
            value: [batch, seq_k, embed_dim]
        Returns:
            out:   [batch, seq_q, embed_dim]
        """
        B, seq_q, _ = query.size()
        _, seq_k, _ = key.size()

        # 1) Project inputs and split into heads
        q = self.q_proj(query).view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_q, head_dim]
        k = self.k_proj(key).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)    # [B, heads, seq_k, head_dim]
        v = self.v_proj(value).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_k, head_dim]

        # 2) Compute scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / self.scale  # [B, heads, seq_q, seq_k]
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        attn = weights @ v  # [B, heads, seq_q, head_dim]

        # 3) Concatenate heads
        attn = attn.transpose(1, 2).contiguous().view(B, seq_q, self.embed_dim)  # [B, seq_q, embed_dim]

        # 4) Final projection
        out = self.out_proj(attn)
        out = self.proj_dropout(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer:
    1) Multi-head self-attention
    2) Feed-forward network
    3) Residual connections + LayerNorm
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1) Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            key_dim=embed_dim,  # self-attention uses same dimension for Q, K, V
            num_heads=num_heads,
            dropout=dropout,
        )

        # 2) Feed-forward network using nn.Sequential
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )

        # 3) LayerNorm and Dropouts for residuals
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, embed_dim]
        Returns:
            Tensor of same shape
        """
        # 1) Self-attention block
        attn_out = self.self_attn(query=x, key=x, value=x)
        x = x + self.attn_dropout(attn_out) # Residual connection
        x = self.norm1(x)

        # 2) Feed-forward block
        ff = self.ffn(x)
        x = x + self.ff_dropout(ff) # Residual connection
        x = self.norm2(x)

        return x
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, embed_dim: int):
        super().__init__()

        # Clone the provided encoder_layer num_layers times
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.norm(x)
        
        return x