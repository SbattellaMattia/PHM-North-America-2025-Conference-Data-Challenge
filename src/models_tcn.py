import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        super().__init__()
        padding = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)

class TCNRegressor(nn.Module):
    """
    Input: [B, L, D] embeddings
    TCN expects [B, D, L]
    Output: [B, 3]
    """
    def __init__(self, in_dim, channels=(128,128,128), kernel_size=5, dropout=0.2, out_dim=3):
        super().__init__()
        layers = []
        ch_in = in_dim
        for i, ch_out in enumerate(channels):
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size, dilation=2**i, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch_in, out_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,D,L]
        h = self.tcn(x)
        y = self.head(h)
        return y
