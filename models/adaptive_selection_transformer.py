import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding (standard sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        S = x.size(1)
        return x + self.pe[:, :S]


# ---------------------------------------------------------------------------
# Multi-head self-attention with per-sample head gating
# ---------------------------------------------------------------------------

class MaskedMultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention where we can apply a *per-sample* head mask mh_l in [0,1]^H.

    Input:
      x: (B, S, d_model)
      head_mask: (B, H) in [0,1] (probabilities; can be "soft" gates)

    Output:
      out: (B, S, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.n_heads
        d_h = self.d_head

        # (B, S, D) -> (B, S, D)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B, S, D) -> (B, H, S, d_h)
        def reshape_heads(t):
            return t.view(B, S, H, d_h).transpose(1, 2)  # (B, H, S, d_h)

        Q = reshape_heads(Q)
        K = reshape_heads(K)
        V = reshape_heads(V)

        # scaled dot-product attention
        # scores: (B, H, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, H, S, S) x (B, H, S, d_h) -> (B, H, S, d_h)
        context = torch.matmul(attn, V)

        # ---- apply head mask (soft gating, per sample, per head) ----
        # head_mask: (B, H) -> (B, H, 1, 1)
        head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
        context = context * head_mask                       # (B, H, S, d_h)

        # merge heads: (B, H, S, d_h) -> (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        out = self.out_proj(context)
        return out


# ---------------------------------------------------------------------------
# Gumbel-sigmoid relaxation for Bernoulli gates
# ---------------------------------------------------------------------------

def gumbel_sigmoid(logits: torch.Tensor,
                   tau: float = 1.0,
                   training: bool = True) -> torch.Tensor:
    """
    Gumbel-Softmax style relaxation for Bernoulli variables.

    Returns y in (0,1) with gradients; at inference we fall back to plain sigmoid.
    """
    if not training:
        return torch.sigmoid(logits)

    # Gumbel noise
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    y = torch.sigmoid((logits + g) / tau)
    return y


# ---------------------------------------------------------------------------
# One transformer block with patch / head / block selection
# ---------------------------------------------------------------------------

class AdaptiveBlock(nn.Module):
    """
    Transformer encoder block with *adaptive* patch, head, and block selection.

    - Patch selection: Mp_l ∈ (0,1)^{B×N} multiplies tokens entering the block.
    - Head selection:  Mh_l ∈ (0,1)^{B×H} gates attention heads.
    - Block selection: Mb_l ∈ (0,1)^B gates the whole block (residual).

    We use soft gates via Gumbel-sigmoid; at inference you can threshold them
    to get hard selections and real compute savings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        num_patches: int,
        dropout: float = 0.1,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_ff = dim_feedforward
        self.num_patches = num_patches
        self.tau = gumbel_tau

        # ---- standard Transformer sublayers ----
        self.self_attn = MaskedMultiheadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        # ---- decision network: three heads Wp_l, Wh_l, Wb_l ----
        # We condition on a pooled representation of Z_l (mean over patches).
        self.dec_patch = nn.Linear(d_model, num_patches)  # mp_l
        self.dec_head = nn.Linear(d_model, n_heads)       # mh_l
        self.dec_block = nn.Linear(d_model, 1)            # mb_l

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, d_model)

        Returns:
          x_out: (B, N, d_model)
          gates: dict with keys "patch", "head", "block" (probabilities)
        """
        B, N, D = x.shape
        H = self.n_heads

        # ---- decision network input Z_l ----
        # In AdaViT this shares previous block outputs; here we use simple mean pooling.
        z_pool = x.mean(dim=1)       # (B, d_model)

        # logits
        mp_logits = self.dec_patch(z_pool)              # (B, N)
        mh_logits = self.dec_head(z_pool)               # (B, H)
        mb_logits = self.dec_block(z_pool).squeeze(-1)  # (B,)

        # Gumbel-Softmax-style relaxed gates in (0,1)
        patch_gate = gumbel_sigmoid(mp_logits, self.tau, self.training)  # (B, N)
        head_gate  = gumbel_sigmoid(mh_logits, self.tau, self.training)  # (B, H)
        block_gate = gumbel_sigmoid(mb_logits, self.tau, self.training)  # (B,)

        # ---- soft patch selection: gate tokens entering attention ----
        x_masked = x * patch_gate.unsqueeze(-1)  # (B, N, D)

        # ---- attention sublayer with head gating ----
        x_norm = self.norm1(x_masked)
        attn_out = self.self_attn(x_norm, head_gate)  # (B, N, D)

        # block_gate: (B,) -> (B,1,1) to gate residual contribution
        bg = block_gate.view(B, 1, 1)
        x = x + bg * self.dropout1(attn_out)

        # ---- feed-forward sublayer ----
        y = self.norm2(x)
        y = self.linear2(self.dropout2(self.activation(self.linear1(y))))
        x = x + bg * y  # gate entire FFN contribution

        gates = {
            "patch": patch_gate,  # (B, N)
            "head": head_gate,    # (B, H)
            "block": block_gate,  # (B,)
        }
        return x, gates


# ---------------------------------------------------------------------------
# FLOPs estimator for a dense (full) block
# ---------------------------------------------------------------------------

def estimate_full_block_flops(num_patches: int,
                              d_model: int,
                              dim_ff: int,
                              n_heads: int) -> float:
    """
    Very rough FLOPs estimate for a single *dense* Transformer block for one sample.

    Returns: scalar (float) FLOPs per sample per block.
    """
    N = num_patches
    D = d_model
    H = n_heads
    d_h = D // H

    # Q/K/V projections and output projection
    flops_qkv = 3.0 * N * D * D
    flops_out = N * D * D

    # attention scores and context
    flops_scores = 2.0 * H * N * N * d_h  # mul+add per score

    # FFN: two linear layers
    flops_ffn = 2.0 * N * D * dim_ff

    return flops_qkv + flops_out + flops_scores + flops_ffn


# ---------------------------------------------------------------------------
# Full model: CNN + adaptive Transformer + FLOPs tracking
# ---------------------------------------------------------------------------

class AdaptiveSelectionCNNTransformer(nn.Module):
    """
    CNN + Transformer for 1D ECG segments with *adaptive* patch, head, and block selection.

    - CNN stem embeds the 1D signal into patch embeddings.
    - Each Transformer block has gates:
        Mp_l : patch (token) gate
        Mh_l : head gate
        Mb_l : block gate
    - We keep a differentiable relaxation using Gumbel-sigmoid.
    - We estimate FLOPs relative to the full dense network so you can log
      the same style of "compute usage" metric as before.
    """

    def __init__(
        self,
        seq_len: int = 5000,
        patch_len: int = 50,
        d_model: int = 128,
        n_heads: int = 2,
        num_layers: int = 4,
        num_classes: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_ff = dim_feedforward

        # ----- CNN patch embedding -----
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len,
        )
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)

        # ----- stack of adaptive transformer blocks -----
        self.blocks = nn.ModuleList([
            AdaptiveBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                num_patches=self.num_patches,
                dropout=dropout,
                gumbel_tau=gumbel_tau,
            )
            for _ in range(num_layers)
        ])

        # ----- MLP classification head -----
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ----- baseline FLOPs (dense) for normalizing -----
        block_flops = estimate_full_block_flops(
            num_patches=self.num_patches,
            d_model=d_model,
            dim_ff=dim_feedforward,
            n_heads=n_heads,
        )
        self.full_block_flops = block_flops          # per-sample per-block
        self.full_model_flops = block_flops * num_layers

    def _cnn_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, seq_len)
        returns: (B, num_patches, d_model)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)          # (B, d_model, num_patches)

        x = x.transpose(1, 2)     # (B, num_patches, d_model)
        x = self.pos_encoder(x)
        return x

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        x: (B, 1, seq_len)

        Returns:
          logits: (B, num_classes)
          compute_fraction: scalar ≈ (mean FLOPs / full_model_flops)  (like a ponder term)
          (optionally) stats: dict with per-batch mean gate usages and flops.
        """
        B = x.size(0)
        device = x.device

        # ----- CNN patch embedding -----
        x = self._cnn_embed(x)  # (B, N, D)
        N = self.num_patches
        D = self.d_model
        H = self.n_heads

        # ----- pass through adaptive blocks -----
        all_patch_gates = []
        all_head_gates = []
        all_block_gates = []

        # FLOPs per-sample accumulator
        flops_per_sample = torch.zeros(B, device=device)

        for block in self.blocks:
            x, gates = block(x)
            Mp = gates["patch"]   # (B, N)
            Mh = gates["head"]    # (B, H)
            Mb = gates["block"]   # (B,)

            all_patch_gates.append(Mp)
            all_head_gates.append(Mh)
            all_block_gates.append(Mb)

            # ----- FLOPs estimation for this block (per sample) -----
            # Effective tokens and heads (soft counts)
            tokens_eff = Mp.sum(dim=1)           # (B,)
            heads_eff = Mh.sum(dim=1)            # (B,)
            # Normalize
            token_ratio = tokens_eff / float(N)  # in [0,1]
            head_ratio = heads_eff / float(H)    # in [0,1]
            block_gate = Mb.clamp(0.0, 1.0)      # (B,)

            # Q/K/V + out_proj scale ~ linearly with #tokens and block gate
            flops_qkv_out = (3.0 * N * D * D + N * D * D)
            flops_qkv_out = flops_qkv_out * token_ratio * block_gate

            # attention scores ~ O(H * N^2 * d_head); scale with (#heads, #tokens^2)
            d_h = D // H
            flops_scores = 2.0 * H * (N ** 2) * d_h
            flops_scores = flops_scores * head_ratio * (token_ratio ** 2) * block_gate

            # FFN ~ O(N * D * dim_ff); scale with #tokens and block gate
            flops_ffn = 2.0 * N * D * self.dim_ff
            flops_ffn = flops_ffn * token_ratio * block_gate

            flops_block = flops_qkv_out + flops_scores + flops_ffn  # (B,)
            flops_per_sample = flops_per_sample + flops_block

        # ----- global representation + head -----
        # simple average over patches (you could switch to a cls token if you wish)
        z = x.mean(dim=1)           # (B, D)
        logits = self.mlp_head(z)   # (B, num_classes)

        # normalize FLOPs to get a "compute fraction" like a ponder term
        # (if gates are all 1, we should be close to 1.0 on average)
        flops_mean = flops_per_sample.mean()
        compute_fraction = flops_mean / (self.full_model_flops + 1e-9)

        if not return_stats:
            # keep a simple 2-tuple pattern (logits, compute-usage-term)
            return logits, compute_fraction

        # ----- aggregate stats for logging -----
        # stack along layer dim: (L, B, N/H) etc.
        Mp_all = torch.stack(all_patch_gates, dim=0)   # (L, B, N)
        Mh_all = torch.stack(all_head_gates, dim=0)    # (L, B, H)
        Mb_all = torch.stack(all_block_gates, dim=0)   # (L, B)

        # mean over layers and batch
        mean_patch_keep = Mp_all.mean().item()
        mean_head_keep = Mh_all.mean().item()
        mean_block_keep = Mb_all.mean().item()

        stats = {
            "flops_per_sample": flops_per_sample.detach().cpu(),   # tensor(B,)
            "flops_mean": float(flops_mean.item()),
            "compute_fraction": float(compute_fraction.item()),
            "mean_patch_keep": mean_patch_keep,
            "mean_head_keep": mean_head_keep,
            "mean_block_keep": mean_block_keep,
        }
        return logits, compute_fraction, stats
