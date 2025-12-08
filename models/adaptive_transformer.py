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
    Adaptive-depth CNN+Transformer for 1D ECG segments.

    - 2 Conv1d layers for patch embedding → (B, num_patches, d_model)
    - Stack of TransformerEncoderLayers
    - ACT-style halting over *depth* (layers), inspired by A-ViT Block_ACT.forward_act
    - Aggregates layer outputs with (delta1 + delta2) (Graves ACT)
    - Returns: logits, ponder_loss

    ponder_loss is the batch-mean "computation time" ρ ≈ (N + r) from ACT.
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
        halt_epsilon: float = 0.05,   # epsilon like self.eps in A-ViT
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

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)

        # ---- explicit stack of transformer layers ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        # ---- halting parameters (similar to gate_scale/gate_center) ----
        # h_l = sigmoid( gamma * z_l[..., 0] - center )
        self.halt_gamma = nn.Parameter(torch.tensor(1.0))
        self.halt_center = nn.Parameter(torch.tensor(1.0))

        # ---- 2-layer MLP head ----
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, return_rho: bool = False):
        """
        Training-time forward with ACT-style depth halting.

        x: (B, 1, seq_len)
        returns: logits (B, num_classes), ponder_loss (scalar), optional rho per sample
        """
        B = x.size(0)
        device = x.device
        L = self.num_layers
        eps = self.halt_epsilon

        # ---------- CNN patch embedding ----------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)          # (B, d_model, num_patches)

        x = x.transpose(1, 2)     # (B, num_patches, d_model)
        x = self.pos_encoder(x)   # add PE

        # ---------- ACT over depth (layers) ----------
        # This mirrors the structure of forward_features_act_token from A-ViT,
        # but with depth as the "time" dimension and a single global token z_l.

        # c: cumulative halting
        c = torch.zeros(B, device=device)          # like c_token
        # R: remainder mass
        R = torch.ones(B, device=device)           # like R_token
        # mask: which samples are still active (1 = active, 0 = done)
        mask = torch.ones(B, device=device)        # like mask_token
        # rho: computation cost (ponder)
        rho = torch.zeros(B, device=device)        # like rho_token

        # output representation (weighted sum over depths)
        output = torch.zeros(B, self.d_model, device=device)

        for l, layer in enumerate(self.layers):
            # Run full layer (we don't bother masking x here; A-ViT does for tokens)
            x = layer(x)                           # (B, S, d_model)
            z = x.mean(dim=1)                      # (B, d_model) global ECG representation

            # ---- halting score for this depth (like halting_score_token) ----
            first_dim = z[:, 0]                    # (B,)
            h_l = torch.sigmoid(self.halt_gamma * first_dim - self.halt_center)  # (B,)

            # Force final layer to halt everyone (like setting h=1 at last)
            if l == L - 1:
                h_l = torch.ones_like(h_l)

            # Only active samples contribute
            active = (mask > 0.0)
            h_eff = h_l * active.float()           # (B,)

            # Update cumulative halting
            c = c + h_eff

            # Base cost: each active sample pays 1 layer compute (like rho_token += mask)
            rho = rho + mask

            # Determine which just reached threshold, which still haven't
            reached = (c > 1.0 - eps) & active     # Case 1 in A-ViT
            not_reached = (c < 1.0 - eps) & active # Case 2 in A-ViT

            reached_f = reached.float()
            not_reached_f = not_reached.float()

            # Case 1: reached in this layer → use remainder R as weight (delta1)
            # delta1 = z * R * reached
            delta1 = z * (R * reached_f).unsqueeze(-1)
            # extra ponder from remainder (rho_token += R * reached)
            rho = rho + R * reached_f

            # Case 2: not reached yet → weight by h_l (delta2)
            # Update remainder: R = R - h_l where not reached
            R = R - (h_eff * not_reached_f)
            delta2 = z * (h_eff * not_reached_f).unsqueeze(-1)

            output = output + delta1 + delta2

            # Update mask: still active if below threshold
            mask = (c < 1.0 - eps).float()

        # After all layers: rho ≈ N + r per sample
        ponder_loss = rho.mean()

        logits = self.mlp_head(output)  # (B, num_classes)
        if return_rho:
            return logits, ponder_loss, rho
        return logits, ponder_loss

    @torch.no_grad()
    def forward_infer(self, x: torch.Tensor, eps: float = None):
        """
        Inference-time forward with *logical* early exit.

        NOTE: This version still runs all layers (so it's simple & backprop-safe),
        but uses the same ACT logic to build 'output'. If you want *true* compute
        savings at inference, we can further mask x per layer and break early.
        """
        if eps is None:
            eps = self.halt_epsilon

        B = x.size(0)
        device = x.device
        L = self.num_layers

        # CNN patch embedding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        c = torch.zeros(B, device=device)
        R = torch.ones(B, device=device)
        mask = torch.ones(B, device=device)

        output = torch.zeros(B, self.d_model, device=device)

        for l, layer in enumerate(self.layers):
            x = layer(x)
            z = x.mean(dim=1)

            first_dim = z[:, 0]
            h_l = torch.sigmoid(self.halt_gamma * first_dim - self.halt_center)
            if l == L - 1:
                h_l = torch.ones_like(h_l)

            active = (mask > 0.0)
            h_eff = h_l * active.float()
            c = c + h_eff

            reached = (c > 1.0 - eps) & active
            not_reached = (c < 1.0 - eps) & active

            reached_f = reached.float()
            not_reached_f = not_reached.float()

            delta1 = z * (R * reached_f).unsqueeze(-1)
            R = R - (h_eff * not_reached_f)
            delta2 = z * (h_eff * not_reached_f).unsqueeze(-1)

            output = output + delta1 + delta2

            mask = (c < 1.0 - eps).float()

            # If you *really* want early break at inference, uncomment:
            # if not mask.any():
            #     break

        logits = self.mlp_head(output)
        return logits
