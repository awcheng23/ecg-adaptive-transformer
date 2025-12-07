import math
import torch
import torch.nn as nn


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


class AdaptiveCNNTransformer(nn.Module):
    """
    Adaptive-depth CNN+Transformer for ECG segments.

    Input:  (B, 1, 5000)
    Steps:
      - 2 Conv1d layers for patch embedding -> (B, 100, d_model)
      - stack of TransformerEncoderLayers
      - ACT-style halting distribution over layers
      - mean-field aggregation over depth -> final representation
      - 2-layer MLP head -> logits (B, 2)
      - returns (logits, ponder_loss)
    """

    def __init__(
        self,
        seq_len: int = 5000,
        patch_len: int = 50,      # 5000 / 50 = 100 patches
        d_model: int = 128,
        n_heads: int = 2,
        num_layers: int = 4,
        num_classes: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        halt_epsilon: float = 0.05,   # epsilon in ACT / A-ViT
    ):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.halt_epsilon = halt_epsilon

        # ---- CNN patch embedding ----
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

        # ---- Positional encoding ----
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)

        # ---- Transformer: explicit stack of layers ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, S, E)
        )
        self.layers = nn.ModuleList(
            [encoder_layer for _ in range(num_layers)]
        )

        # ---- Halting module parameters (shared across layers) ----
        # h_l = sigmoid( gamma * state_l[..., 0] + beta )
        self.halt_gamma = nn.Parameter(torch.tensor(5.0))
        self.halt_beta  = nn.Parameter(torch.tensor(-10.0))

        # ---- 2-layer MLP classification head ----
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),   # logits for 0/1
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, 5000)
        returns: logits (B, 2), ponder_loss (scalar)
        """
        B = x.size(0)

        # ---- CNN patch embedding ----
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)          # (B, d_model, 100)

        # (B, 100, d_model)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        # ---- run transformer layer-by-layer and store global states ----
        states = []  # list of length L, each (B, d_model)

        for layer in self.layers:
            x = layer(x)                  # (B, S, d_model)
            # Global representation at this depth: mean over patches
            states.append(x.mean(dim=1))  # (B, d_model)

        # Stack to shape (B, L, d_model)
        states_stack = torch.stack(states, dim=1)
        B, L, D = states_stack.shape
        device = states_stack.device

        # ---- halting scores h_l (per layer, per sample) ----
        # Use the first embedding dimension as in A-ViT (e = 0) :contentReference[oaicite:1]{index=1}
        first_dim = states_stack[..., 0]          # (B, L)
        h = torch.sigmoid(self.halt_gamma * first_dim + self.halt_beta)  # (B, L)

        # Enforce halting at final layer: h_L = 1  :contentReference[oaicite:2]{index=2}
        h[:, -1] = 1.0

        eps = self.halt_epsilon

        # ---- ACT-style halting distribution over layers ----
        cumul = torch.zeros(B, device=device)     # cumulative halting score
        R     = torch.ones(B, device=device)      # remainder
        ponder = torch.zeros(B, device=device)    # ρ (N + r) per sample
        p = torch.zeros(B, L, device=device)      # halting distribution over layers

        for l in range(L):
            h_l = h[:, l]                         # (B,)

            # which samples are still "running" before this layer?
            running = cumul < (1.0 - eps)         # (B,) boolean
            h_eff = h_l * running.float()         # effective h only where running

            # add 1 per active layer (like ρ += m in Alg. 1) :contentReference[oaicite:3]{index=3}
            ponder = ponder + running.float()

            R_prev = R.clone()

            # update cumulative halting
            cumul = cumul + h_eff

            # which samples are still running after this layer?
            running_new = cumul < (1.0 - eps)
            just_halted = running & (~running_new)

            # update remainder: R <- R - h_l where still running
            R = torch.where(running, R - h_eff, R)

            # build halting probability p_l  (single-token version of Eq. 7) :contentReference[oaicite:4]{index=4}
            p_l = torch.zeros(B, device=device)
            # if l < N: p_l = h_l
            p_l = torch.where(running_new, h_eff, p_l)
            # if l == N: p_l = R_prev
            p_l = torch.where(just_halted, R_prev, p_l)

            p[:, l] = p_l

        # add remainder r to ponder: ρ = N + r (Eq. 8) :contentReference[oaicite:5]{index=5}
        ponder = ponder + R
        ponder_loss = ponder.mean()  # scalar over batch

        # ---- mean-field aggregation over depth (Eq. 9, adapted) :contentReference[oaicite:6]{index=6}
        # to = sum_l p_l * state_l
        out_rep = (p.unsqueeze(-1) * states_stack).sum(dim=1)  # (B, d_model)

        logits = self.mlp_head(out_rep)  # (B, 2)
        return logits, ponder_loss
