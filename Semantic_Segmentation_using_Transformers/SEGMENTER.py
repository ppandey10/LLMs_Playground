# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
device

#==============================================================================================#
# Patch embedding class
class PatchEmbedding(nn.Module):
    """
    Turns 2D images into patches and then flattens each patch.
    """
    def __init__(self, in_channels: int, patch_size: int, embd_dim: int):
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels=in_channels, # 3
            out_channels=embd_dim, # 768, since each patch 16x16 with 3 channels. Hence, each patch should have 768 tokens.
            kernel_size=patch_size, # 16
            stride=patch_size, # 16
        )

        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # (b, c, h, w) -> (b, c, h*w)
    
    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1) # (b, embd_dim, num_patches) -> (b, num_patches, embd_dim)
    
#================================================================================================#
# Multihead attention block
class MultiHeadAttentionBlock(nn.Module):
    """
    Creates multihead attention blocks
    """
    def __init__(
        self,
        embd_dim: int, 
        num_heads: int,
        attn_drop: float
    ):
        super().__init__()

        # Define normalization layer (mentioned in the ViT paper)
        self.layernorm = nn.LayerNorm(normalized_shape=embd_dim)

        # Define multihead attention from torch library
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embd_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True, # makes sure that batch dimension comes first
        )

    def forward(self, x):
        x = self.layernorm(x)
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x, 
            need_weights=False, # do we need the weights or just the layer outputs?
        )
        return attn_output
    
#================================================================================================#
# Define MLP block 
class MLPBlock(nn.Module):
    """
    Simple MLP block from the ViT paper
    """
    def __init__(
        self, 
        embd_dim: int,
        mlp_size: int, # 3072 (as mentioned in ViT paper)
        mlp_drop: float,
    ):
        super().__init__()
        # Normalization
        self.layernorm = nn.LayerNorm(normalized_shape=embd_dim)

        # Feed forward network 
        self.ffwd = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(in_features=mlp_size, out_features=embd_dim),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.layernorm(x)
        x = self.ffwd(x)
        return x
    
#================================================================================================#
# Define Encoder
class SegmenterEncoder(nn.Module):
    """
    Encoder block of Segmenter with residual connections. It's structure is similar to ViT.
    """
    def __init__(
        self, 
        embd_dim: int, 
        num_heads: int,
        mlp_size: int, 
        attn_drop: float, 
        mlp_drop: float
    ):
        super().__init__()
        # start with multihead self attention
        self.msa = MultiHeadAttentionBlock(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_drop=attn_drop
        )

        self.mlp = MLPBlock(
            embd_dim=embd_dim,
            mlp_size=mlp_size,
            mlp_drop=mlp_drop
        )

    def forward(self, x):
        # add residual connections
        x += self.msa(x)
        x += self.mlp(x)
        return x