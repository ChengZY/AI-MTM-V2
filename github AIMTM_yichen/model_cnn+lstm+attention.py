import torch
import torch.nn as nn


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size, cnn_channels=64, cnn_kernel_size=7,
                 cnn_stride=4, cnn_layers=2, hidden_size=128, num_layers=2,
                 output_size=7, num_heads=8, dropout=0.1, bidirectional=False):
        super(CNN_LSTM_Attention, self).__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.num_directions = 2 if bidirectional else 1

        cnn_block = []
        in_ch = 1
        for i in range(cnn_layers):
            stride = cnn_stride if i == 0 else 1   # only compress on first layer
            padding = cnn_kernel_size // 2          # same padding to keep edges
            cnn_block += [
                nn.Conv1d(in_ch, cnn_channels,
                          kernel_size=cnn_kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = cnn_channels
        self.cnn_encoder = nn.Sequential(*cnn_block)

        # ── LSTM ──────────────────────────────────────────────────────────────
        # Input to LSTM is cnn_channels (one value per CNN feature map per timestep)
        self.lstm = nn.LSTM(
            input_size   = cnn_channels,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0,
            bidirectional= bidirectional,
        )

        # ── Multi-head Self-Attention ─────────────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim   = hidden_size * self.num_directions,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(hidden_size * self.num_directions)

        # ── Output head ───────────────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_size) — flat spectrum, e.g. (64, 801)

        Returns:
            out: (batch, output_size) — predicted thicknesses
        """
        # Add channel dim for CNN: (batch, 1, input_size)
        x = x.unsqueeze(1)

        # CNN encoding: (batch, cnn_channels, compressed_len)
        x = self.cnn_encoder(x)

        # Transpose for LSTM: (batch, compressed_len, cnn_channels)
        x = x.transpose(1, 2)

        # LSTM: (batch, compressed_len, hidden * directions)
        lstm_out, _ = self.lstm(x)

        # Self-attention over all LSTM timesteps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection + layer norm
        out = self.norm(lstm_out + attn_out)

        # Global average pool across compressed sequence
        out = out.mean(dim=1)   # (batch, hidden * directions)

        return self.fc(out)
