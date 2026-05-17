"""
MultiScale_CNN_LSTM_Attention — add this to your model.py

Architecture (Approach 2 — parallel multi-scale CNN):

    Input spectrum (batch, num_wavelengths)
            ↓
    ┌─────────────────────────────────────────┐
    │  Small kernel CNN  (kernel=5)           │  ← captures tight fringes (200-400nm region)
    │  Medium kernel CNN (kernel=11)          │  ← captures medium fringes
    │  Large kernel CNN  (kernel=21)          │  ← captures wide fringes (700-1000nm region)
    └──────────────┬──────────────────────────┘
                   │ concatenate along channel dim
                   ↓
    LSTM — reads the multi-scale feature sequence
                   ↓
    Multi-head Self-Attention — global comparison
                   ↓
    Output head → 7 thickness values

Why this works better than single-kernel CNN-LSTM-Attention:
    Your spectrum has a chirped interference pattern — tight fringes at low wavelengths,
    wide fringes at high wavelengths. A single kernel size is always a compromise.
    Running three kernel sizes in parallel lets the model pick up fringe patterns
    at all scales simultaneously, then the LSTM learns how these multi-scale features
    evolve across the spectrum.
"""
import torch
import torch.nn as nn


class MultiScale_CNN_LSTM_Attention(nn.Module):
    """
    Multi-scale CNN encoder + LSTM + Attention for spectral thickness prediction.

    Args:
        input_size      : number of input wavelength points (e.g. 801)
        cnn_channels    : number of output channels PER scale
                          total channels into LSTM = cnn_channels * 3
                          recommended: 32 or 64
        cnn_stride      : compression stride applied to all CNN branches
                          recommended: 4 (same as original CNN-LSTM-Attention)
        cnn_layers      : number of stacked conv layers per branch
                          recommended: 2
        hidden_size     : LSTM hidden size
                          recommended: 128 or 256
        num_layers      : number of stacked LSTM layers
                          recommended: 3
        output_size     : number of thickness predictions (7)
        num_heads       : attention heads
                          NOTE: hidden_size must be divisible by num_heads
                          recommended: 8
        dropout         : dropout rate, recommended: 0.1
        bidirectional   : if True, LSTM reads both directions
    """
    def __init__(self, input_size, cnn_channels=64, cnn_stride=4, cnn_layers=2,
                 hidden_size=256, num_layers=3, output_size=7,
                 num_heads=8, dropout=0.1, bidirectional=False):
        super(MultiScale_CNN_LSTM_Attention, self).__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.num_directions = 2 if bidirectional else 1

        # ── Three parallel CNN branches, each with a different kernel size ────
        # Small kernel — best at detecting tight, high-frequency fringes (200-400nm)
        self.cnn_small  = self._make_cnn(cnn_channels, kernel_size=5,  stride=cnn_stride, layers=cnn_layers, dropout=dropout)
        # Medium kernel — balanced fringe detection (400-700nm)
        self.cnn_medium = self._make_cnn(cnn_channels, kernel_size=11, stride=cnn_stride, layers=cnn_layers, dropout=dropout)
        # Large kernel — best at detecting wide, low-frequency fringes (700-1000nm)
        self.cnn_large  = self._make_cnn(cnn_channels, kernel_size=21, stride=cnn_stride, layers=cnn_layers, dropout=dropout)

        # Total channels after concatenation = cnn_channels * 3 (one per branch)
        lstm_input_size = cnn_channels * 3

        # ── LSTM reads the concatenated multi-scale features ──────────────────
        self.lstm = nn.LSTM(
            input_size    = lstm_input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional,
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

    def _make_cnn(self, out_channels, kernel_size, stride, layers, dropout):
        """Build a CNN branch with `layers` Conv1d blocks."""
        blocks = []
        in_ch = 1
        for i in range(layers):
            s       = stride if i == 0 else 1   # compress only on first layer
            padding = kernel_size // 2           # same padding
            blocks += [
                nn.Conv1d(in_ch, out_channels, kernel_size=kernel_size,
                          stride=s, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: (batch, input_size) — flat spectrum e.g. (64, 801)

        Returns:
            out: (batch, output_size) — predicted thicknesses
        """
        # Add channel dim for CNN: (batch, 1, input_size)
        x = x.unsqueeze(1)

        # Run all three CNN branches in parallel
        # Each produces: (batch, cnn_channels, compressed_len)
        out_small  = self.cnn_small(x)
        out_medium = self.cnn_medium(x)
        out_large  = self.cnn_large(x)

        # Align sequence lengths (may differ slightly due to padding/kernel size)
        # Trim to the shortest sequence length across all three branches
        min_len = min(out_small.size(2), out_medium.size(2), out_large.size(2))
        out_small  = out_small[:, :, :min_len]
        out_medium = out_medium[:, :, :min_len]
        out_large  = out_large[:, :, :min_len]

        # Concatenate along channel dim: (batch, cnn_channels*3, compressed_len)
        x = torch.cat([out_small, out_medium, out_large], dim=1)

        # Transpose for LSTM: (batch, compressed_len, cnn_channels*3)
        x = x.transpose(1, 2)

        # LSTM: (batch, compressed_len, hidden * directions)
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)

        # Global average pool: (batch, hidden * directions)
        out = out.mean(dim=1)

        return self.fc(out)
