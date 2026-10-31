"""CNN-BiLSTM-CTC models for variable-length IMU sequences."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

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


class _ChannelLayerNorm(nn.Module):
    """Normalize channels independently at every timestep."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualBlock1d(nn.Module):
    """Two-convolution residual block with padding-independent normalization."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.norm1 = _ChannelLayerNorm(output_channels)
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = _ChannelLayerNorm(output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride)
        )
        self.activation = nn.GELU()

    def forward(
        self, values: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_lengths = torch.div(
            lengths + self.stride - 1, self.stride, rounding_mode="floor"
        )
        residual = self.skip(values)
        values = self.activation(self.norm1(self.conv1(values)))
        values = _mask_temporal_padding(values, output_lengths)
        values = self.activation(self.norm2(self.conv2(values)) + residual)
        return values, output_lengths


class ResNetBiLSTMCTC(nn.Module):
    """Residual 1D CNN-BiLSTM-CTC recognizer for six-axis IMU sequences."""

    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 6,
        channels: Sequence[int] = (64, 128, 256),
        hidden_size: int = 256,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        if num_classes < 2:
            raise ValueError("num_classes must include at least one label and CTC blank")
        if input_channels < 1 or len(channels) != 3 or any(c < 1 for c in channels):
            raise ValueError("ResNet requires positive input channels and three channel widths")
        if hidden_size < 1 or lstm_layers < 1:
            raise ValueError("hidden_size and lstm_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=7, stride=2, padding=3,
                      bias=False),
            _ChannelLayerNorm(channels[0]),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock1d(channels[0], channels[0]),
                ResidualBlock1d(channels[0], channels[1], stride=2),
                ResidualBlock1d(channels[1], channels[2]),
            ]
        )
        self.recurrent = nn.LSTM(
            input_size=channels[2],
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def encode_temporal(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-timestep BiLSTM features before the CTC projection head."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_channels:
            raise ValueError(
                f"inputs must have shape (batch, time, {self.input_channels})"
            )
        if lengths.ndim != 1 or len(lengths) != len(inputs):
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(lengths < 1) or torch.any(lengths > inputs.shape[1]):
            raise ValueError("input lengths must be within the padded time dimension")

        lengths = lengths.to(dtype=torch.long)
        first_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        output_lengths = encoded_length(lengths)
        features = self.stem(inputs.transpose(1, 2))
        features = _mask_temporal_padding(features, first_lengths)
        stage_lengths = first_lengths
        for block in self.residual_blocks:
            features, stage_lengths = block(features, stage_lengths)
            features = _mask_temporal_padding(features, stage_lengths)

        features = features.transpose(1, 2)
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
        return recurrent, output_lengths

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, output_lengths = self.encode_temporal(inputs, lengths)
        return self.classifier(self.dropout(features)), output_lengths


def _mask_temporal_padding(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    steps = torch.arange(features.shape[-1], device=features.device)
    return features.masked_fill(steps[None, None, :] >= lengths[:, None, None], 0.0)


MODEL_TYPES = ("cnn_bilstm_ctc", "resnet_bilstm_ctc")


def default_model_config(model_type: str, num_classes: int) -> dict:
    """Return the fixed, documented starting configuration for an architecture."""
    if model_type == "cnn_bilstm_ctc":
        return {
            "num_classes": num_classes,
            "input_channels": 6,
            "conv_channels": 32,
            "hidden_size": 64,
            "lstm_layers": 2,
            "dropout": 0.1,
        }
    if model_type == "resnet_bilstm_ctc":
        return {
            "num_classes": num_classes,
            "input_channels": 6,
            "channels": [64, 128, 256],
            "hidden_size": 256,
            "lstm_layers": 3,
            "dropout": 0.2,
        }
    raise ValueError(f"unknown model type {model_type!r}; choose from {MODEL_TYPES}")


def build_model(model_type: str, config: Mapping) -> nn.Module:
    """Construct a model from its checkpointed architecture name and config."""
    if model_type == "cnn_bilstm_ctc":
        return CNNBiLSTMCTC(**dict(config))
    if model_type == "resnet_bilstm_ctc":
        return ResNetBiLSTMCTC(**dict(config))
    raise ValueError(f"unknown model type {model_type!r}; choose from {MODEL_TYPES}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for reproducible architecture comparisons."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ctc_greedy_decode(
    tokens: Sequence[int],
    blank_index: int = 59,
    charset: str | None = None,
) -> str:
    """Collapse repeated CTC tokens and remove blanks using a given charset."""
    if charset is None:
        from smartpen.hwr.data import ONHW_CHARSET

        charset = ONHW_CHARSET
    if blank_index != len(charset):
        raise ValueError("CTC blank index must follow the last charset symbol")

    characters: list[str] = []
    previous: int | None = None
    for raw_token in tokens:
        token = int(raw_token)
        if token != previous and token != blank_index:
            if token < 0 or token >= len(charset):
                raise ValueError(f"CTC token out of range: {token}")
            characters.append(charset[token])
        previous = token
    return "".join(characters)
