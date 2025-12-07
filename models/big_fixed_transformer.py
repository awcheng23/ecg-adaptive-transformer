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


class CNNTransformer(nn.Module):
    """
    Input: (B, 1, 5000) ECG segments

    Changes vs original:
      - Smaller patches (default patch_len=25) → 200 tokens instead of 100
      - Stronger CNN front-end (more channels)
      - Larger Transformer (d_model=256, n_heads=8, num_layers=6 by default)
    """

    def __init__(
        self,
        seq_len: int = 5000,
        patch_len: int = 25,        # 5000 / 25 = 200 patches (more tokens)
        d_model: int = 256,         # must be divisible by n_heads
        n_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model

        # ---- Stronger 2-layer CNN patch embedding ----
        # Conv1: more channels, local feature extraction, length ~5000
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,        # was 32
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()

        # Conv2: patchify (kernel = stride = patch_len)
        # Output: (B, d_model, num_patches)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=d_model,   # larger embedding dim
            kernel_size=patch_len,
            stride=patch_len,
        )
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()

        # ---- Positional encoding & Transformer ----
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, S, E)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---- 2-layer MLP classification head ----
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 5000)
        returns: logits (B, num_classes)
        """
        # CNN patch embedding
        x = self.conv1(x)      # (B, 64, 5000)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)      # (B, d_model, num_patches = 200)
        x = self.bn2(x)
        x = self.act2(x)

        x = x.transpose(1, 2)  # (B, 200, d_model)
        x = self.pos_encoder(x)

        x = self.transformer(x)  # (B, 200, d_model)

        # pool over patches
        x = x.mean(dim=1)        # (B, d_model)

        logits = self.mlp_head(x)
        return logits
