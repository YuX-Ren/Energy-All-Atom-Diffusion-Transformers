"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from: https://github.com/facebookresearch/DiT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine poDiTional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: embedding dimension
        max_len: maximum length

    Returns:
        poDiTional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def modulate(x, shift, scale):
    # TODO this is global modulation; explore per-token modulation
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# rewrite the multihead attention instead of import from torch.nn
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0, bias=True, batch_first=True):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = dropout

    def init_weights(self):
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.q.bias, 0)
        nn.init.constant_(self.k.bias, 0)
        nn.init.constant_(self.v.bias, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        # add key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.to(query.device)
            key_padding_mask = key_padding_mask.bool()
            
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        
        # Optimized attention computation with better memory layout
        batch_size, seq_len, hidden_dim = q.shape
        head_dim = hidden_dim // self.num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Use scaled dot-product attention with optional key padding mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        attn_output = self.out(attn_output)
        return attn_output


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT for energy-based velocity prediction in flow matching."""

    def __init__(self, hidden_dim, out_dim, num_classes):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # Energy head for each class - velocity will be computed as gradient of energy
        self.energy_head = nn.Linear(hidden_dim, num_classes, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # Output energy scores - velocity will be computed as gradient
        energy_scores = self.energy_head(x)
        return energy_scores


class DiT(nn.Module):
    """Joint energy-based flow matching model with a Transformer backbone.

    Args:
        d_x (int): Input dimension
        d_model (int): Model dimension
        num_layers (int): Number of Transformer layers
        nhead (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden to input dimension in MLP
        class_dropout_prob (float): Probability of dropping class labels for classifier-free guidance
        num_datasets (int): Number of datasets for classifier-free guidance
        num_spacegroups (int): Number of spacegroups for classifier-free guidance
    """

    def __init__(
        self,
        d_x=8,
        d_model=384,
        num_layers=12,
        nhead=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_datasets=2,  # clf-free guidance input
        num_spacegroups=230,  # clf-free guidance input
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead
        self.num_datasets = num_datasets

        self.x_embedder = nn.Linear(d_x, d_model, bias=True)
        # self.t_embedder = TimestepEmbedder(d_model)
        # self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x, num_datasets)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        # nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Initialize energy head with small weights
        nn.init.normal_(self.final_layer.energy_head.weight, std=0.02)
        nn.init.constant_(self.final_layer.energy_head.bias, 0)

    def forward(self, x, t, dataset_idx, spacegroup, mask, x_sc=None):
        """Forward pass of joint energy-based DiT.

        Args:
            x (torch.Tensor): Input data tensor (B, N, d_in)
            t (torch.Tensor): Time step for each sample (B,)
            dataset_idx (torch.Tensor): Dataset index for each sample (B,)
            spacegroup (torch.Tensor): Spacegroup index for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
            x_sc (torch.Tensor): Self-conditioning x (B, N, d_in)
            
        Returns:
            torch.Tensor: Energy scores for each dataset class (B, N, num_datasets)
        """
        # Positonal embedding
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)

        # Input embeddings: (B, N, d) - no self-conditioning
        x = self.x_embedder(x) + pos_emb

        # Conditioning embeddings
        # desiable timestep embedding
        # t = self.t_embedder(t.squeeze(1))  # (B, d)
        # d = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        # c = t + d + s  # (B, 1, d)
        c = s 

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer - outputs energy scores
        energy_scores = self.final_layer(x, c)  # (B, N, num_datasets)
        energy_scores = energy_scores * mask[..., None]
        energy_scores = energy_scores.mean(dim=1)
        return energy_scores

    def get_velocity(self, x, t, dataset_idx, spacegroup, mask):
        """Get velocity as gradient of energy with respect to input x.
        
        Args:
            x (torch.Tensor): Input data tensor (B, N, d_in)
            t (torch.Tensor): Time step for each sample (B,)
            dataset_idx (torch.Tensor): Dataset index for each sample (B,)
            spacegroup (torch.Tensor): Spacegroup index for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
            
        Returns:
            torch.Tensor: Velocity field (B, N, d_in)
        """
        x.requires_grad_(True)
        
        # Get energy scores
        energy_scores = self.forward(x, t, dataset_idx, spacegroup, mask)
        
        # Compute velocity as negative gradient of energy
        # For flow matching, we want velocity to point towards lower energy
        velocity = -torch.autograd.grad(
            energy_scores.sum(), x, create_graph=True, retain_graph=True
        )[0]
        
        return velocity

    def get_velocity_and_logits(self, x, t, dataset_idx, spacegroup, mask):
        """Get velocity as gradient of energy with respect to input x and logits for each dataset class.
        
        Args:
            x (torch.Tensor): Input data tensor (B, N, d_in)
            t (torch.Tensor): Time step for each sample (B,)
            dataset_idx (torch.Tensor): Dataset index for each sample (B,)
            spacegroup (torch.Tensor): Spacegroup index for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
            
        Returns:
            torch.Tensor: Velocity field (B, N, d_in)
            torch.Tensor: Logits for each dataset class (B, N, num_datasets)
        """
        with torch.enable_grad():   
            x.requires_grad_(True)
        
            # Get energy scores
            energy_scores = self.forward(x, t, dataset_idx, spacegroup, mask)
            # energy equals log-sum-exp of energy_scores
            energy = torch.logsumexp(energy_scores, dim=1)
            
            # Compute velocity as negative gradient of energy
            velocity = -torch.autograd.grad(
                energy, x, grad_outputs=torch.ones(energy.shape[0], device=energy.device), create_graph=True, retain_graph=True,
            )[0]
        return velocity, energy_scores

    def forward_with_cfg(self, x, t, dataset_idx, spacegroup, mask, cfg_scale):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free
        guidance.

        Assumes batch x's and class labels are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.
        """
        # compute energy score over the dataset_idx
        with torch.enable_grad():   
            x.requires_grad_(True)
            single_energy_score = self.forward(x, t, dataset_idx, spacegroup, mask)
            # energy scores shape (B, num_datasets) dataset_idx is (B,)
            # single_energy_score is (B,1)
            single_energy_score = single_energy_score.gather(1, dataset_idx.unsqueeze(1)).squeeze(1)
            # compute velocity as gradient of energy
            try:
                velocity = -torch.autograd.grad(
                    single_energy_score, x, grad_outputs=torch.ones(single_energy_score.shape[0], device=single_energy_score.device), create_graph=False, retain_graph=False,
                )[0]
            except Exception as e:
                raise ValueError(single_energy_score.shape)
        del single_energy_score
        velocity = velocity.detach()
        return velocity
