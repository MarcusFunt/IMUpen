"""Compact CNN-BiLSTM acoustic model with a CTC character head."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def encoded_length(lengths: int | Sequence[int] | torch.Tensor):
    """Output lengths after two kernel-5, stride-2, padding-2 convolutions."""
    if isinstance(lengths, torch.Tensor):
        return torch.div(lengths + 3, 4, rounding_mode="floor")
    if isinstance(lengths, int):
        return (lengths + 3) // 4
    return [(int(length) + 3) // 4 for length in lengths]


class CNNBiLSTMCTC(nn.Module):
    """Map a variable-length (batch, time, six axes) sequence to CTC logits."""

    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 6,
        conv_channels: int = 64,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include at least one label and CTC blank")
        self.conv1 = nn.Conv1d(
            input_channels, conv_channels, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv1d(
            conv_channels, conv_channels * 2, kernel_size=5, stride=2, padding=2
        )
        self.activation = nn.GELU()
        self.recurrent = nn.LSTM(
            input_size=conv_channels * 2,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != 6:
            raise ValueError("inputs must have shape (batch, time, 6)")
        if lengths.ndim != 1 or len(lengths) != len(inputs):
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(lengths < 1) or torch.any(lengths > inputs.shape[1]):
            raise ValueError("input lengths must be within the padded time dimension")

        features = self.activation(self.conv1(inputs.transpose(1, 2)))
        first_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        first_steps = torch.arange(features.shape[-1], device=features.device)
        features = features.masked_fill(
            first_steps[None, None, :] >= first_lengths[:, None, None], 0.0
        )
        features = self.activation(self.conv2(features))
        output_lengths = encoded_length(lengths.to(dtype=torch.long))
        output_steps = torch.arange(features.shape[-1], device=features.device)
        features = features.masked_fill(
            output_steps[None, None, :] >= output_lengths[:, None, None], 0.0
        ).transpose(1, 2)
        packed = pack_padded_sequence(
            features,
            output_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        recurrent, _ = self.recurrent(packed)
        recurrent, _ = pad_packed_sequence(
            recurrent, batch_first=True, total_length=features.shape[1]
        )
        return self.classifier(recurrent), output_lengths


def ctc_greedy_decode(tokens: Sequence[int], blank_index: int = 59) -> str:
    """Collapse consecutive CTC tokens and remove blanks using OnHW charset."""
    from smartpen.hwr.data import ONHW_CHARSET

    characters: list[str] = []
    previous: int | None = None
    for raw_token in tokens:
        token = int(raw_token)
        if token != previous and token != blank_index:
            if token < 0 or token >= len(ONHW_CHARSET):
                raise ValueError(f"CTC token out of range: {token}")
            characters.append(ONHW_CHARSET[token])
        previous = token
    return "".join(characters)
