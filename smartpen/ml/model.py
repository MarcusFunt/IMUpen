"""Variable-length CNN-BiLSTM-CTC handwriting baseline.

Two stride-2 convolutions yield ceil(T/4) frames. Masking padding after
each convolution makes batch composition irrelevant at sequence boundaries.
"""
from __future__ import annotations
import torch
from torch import nn

def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    return (lengths + 3) // 4

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes: int, *, channels: int = 64,
                 hidden_size: int = 128, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        if num_classes < 2 or channels < 4 or hidden_size < 2 or layers < 1:
            raise ValueError("Invalid model dimensions")
        self.config = dict(num_classes=num_classes, channels=channels,
                           hidden_size=hidden_size, layers=layers, dropout=dropout)
        self.cnn = nn.Sequential(
            nn.Conv1d(6, channels, 7, stride=2, padding=3, bias=False),
            nn.Identity(), nn.GELU(),
            nn.Conv1d(channels, channels * 2, 5, stride=2, padding=2, bias=False),
            nn.Identity(), nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, 3, padding=1, bias=False),
            nn.Identity(), nn.GELU(),
        )
        self.rnn = nn.LSTM(channels * 2, hidden_size, num_layers=layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != 6:
            raise ValueError("Input must be [B,T,6]")
        if lengths.ndim != 1 or len(lengths) != x.shape[0]:
            raise ValueError("Input lengths must match batch size")
        time = torch.arange(x.shape[1], device=x.device)
        activations = x.transpose(1, 2).masked_fill(
            (time[None, :] >= lengths.to(x.device)[:, None])[:, None, :], 0)
        current_lengths = lengths.to(x.device)
        for layer in self.cnn:
            activations = layer(activations)
            if isinstance(layer, nn.Conv1d):
                if layer.stride[0] == 2:
                    current_lengths = (current_lengths + 1) // 2
                valid = torch.arange(activations.shape[-1], device=x.device)
                activations = activations.masked_fill(
                    (valid[None, :] >= current_lengths[:, None])[:, None, :], 0)
        features = activations.transpose(1, 2)
        out_lengths = output_lengths(lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            features, out_lengths.detach().to("cpu"),
            batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=features.shape[1])
        return self.head(encoded).log_softmax(-1).transpose(0, 1), out_lengths

def required_ctc_steps(label: list[int] | tuple[int, ...]) -> int:
    return len(label) + sum(label[i] == label[i - 1] for i in range(1, len(label)))
