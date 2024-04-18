# import necessary libraries
import numpy as  np 
import time 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#=======================================================================================================#
# Dataloaders

#=======================================================================================================#
# Highlight of Vision Transformers: Image splitting and embedding them!
class PatchEmbedding(nn.Module):
    """
    Splits images into patches, embed and flatten them as 1D vector!
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 # no of patches

        # splitting images into patches 
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Run forward pass. 
        To feed my images into the transformer. I have to flatten out each patch with shape (196, 768) where 196 is total number of patches and 768 is embedding dimension.
        """
        # x: (n_samples, in_channels, img_size, img_size)
        x = self.proj(x) # (n_samples, embd_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2) # (n_samples, embd_dim, num_patches)
        # swap two dimensions
        x = x.transpose(1,2) # (n_samples, num_patches, embd_dim)
        return x

#=======================================================================================================#
# Create the Attention module
class AttentionModule(nn.Module):
    """
    Block to implement the attention mechanism 
    """
    def __init__(self, dim, n_heads=10, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim # channels
        self.head_dim = (dim // n_heads)

        # creating query, key and value for each token in a different manner
        self.qkv = nn.Linear(
            in_features=dim, 
            out_features=dim*3, 
            bias=qkv_bias
        )
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Run forward pass
        """
        n_samples, num_tokens, dim = x.shape

        # sanity check
        if dim != self.dim:
            raise ValueError
        
        # extract query, key, valu
        qkv = self.qkv(x) # (n_samples, num_patches + 1, 3*dim)
        # create extra dimension for num_heads and key, query, value
        qkv = qkv.reshape(
            n_samples, num_tokens, 3, self.n_heads, self.head_dim
        )
        # change the order
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, num_heads, num_tokens, head_dim)
        # extract q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]
        # calculate the attention score (weighted average)
        wei_avg = q @ k.transpose(-2, -1) * dim ** (-0.5)
        # apply softmax for normalisation 
        wei_avg = F.softmax(wei_avg, dim=-1)
        wei_avg = self.attn_drop(wei_avg)
        
        # for value
        out = wei_avg @ v
        # swapping 
        out = out.transpose(1, 2)
        out = out.flatten(2)
        # last two operations concatenated the results from each head
        out = self.proj(out)
        out = self.proj_drop(out)
        return out 
    
#========================================================================================================#
# Create the feedforward network or MLP
class FeedForwardNet(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        )

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)

#=========================================================================================================#
# Tranformer Block
class VTransformerBlock(nn.Module):
    """
    Individual encoder block for the vision transformer
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.): # dim: embedding dimension
        super().__init__()
        self.self_attn = AttentionModule(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p, 
            proj_p=p
        )
        self.ffwd = FeedForwardNet(
            in_features=dim, 
            hidden_features=mlp_ratio*dim, 
            out_features=dim
        )
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Run forward pass
        """
        x = x + self.self_attn(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2)
        return x
    
#======================================================================================================#
# Create Vision Transformer
class VisionTransformer(nn.Module):
    """
    Scratch implementation of Vision Transformers
    """
    def __init__(
            self, 
            img_size=384,
            patch_size=16, # number of patches = 24 (for img: 384 and patch: 16)
            in_chans=3,
            n_classes=1000,
            embed_dim=768, # each patch is 16x16 with 3 channel. therefore, 16x16x3=768
            depth=12, # no of encoders
            n_heads=12,
            qkv_bias=True,
            mlp_ratio=4.0,
            attn_p=0.,
            p=0.
        ):
        super().__init__()

        # start with patch embedding 
        self.patch_embd = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # include the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # position embedding
        self.pos_embd = nn.Parameter(torch.zeros(1, 1 + self.patch_embd.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
            VTransformerBlock(
                dim=embed_dim, 
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                p=p
            )
            for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        # Linear projection to get the probablities for each class
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embd(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x