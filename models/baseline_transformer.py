import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGTransformer(nn.Module):
    """
    A simple transformer classifier for ECG signals.
    Input shape: (batch, seq_len)
    Outputs: class logits
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Project 1D ECG input into model dimension
        self.input_proj = nn.Linear(1, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len) or (batch, 1, seq_len)
        """

        # ensure shape = (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.transpose(1, 2)

        # project to d_model
        x = self.input_proj(x)

        # add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]

        # transformer layers
        x = self.transformer(x)

        # use CLS = mean-pooling over time
        pooled = x.mean(dim=1)

        # final classifier
        return self.cls_head(pooled)
