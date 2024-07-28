from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn


class LSTMDropout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, hidden_out = self.lstm(x, hidden_in)
        return self.dropout(x), hidden_out


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(input_size, 4 * hidden_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RecurrentResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(input_dim)
        self.lstm = LSTMDropout(input_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, hidden_out = self.lstm(self.ln_1(x), hidden_in)
        x = x + self.mlp(self.ln_2(x))
        return x, hidden_out
