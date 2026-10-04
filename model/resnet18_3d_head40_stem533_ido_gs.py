"""Independent Head40-533 ResNet18 backbone for the IDO-GS experiment.

The network consumes float volumes shaped ``[batch, 1, depth, height, width]``
and returns unnormalised class logits.  Cropping and head-direction alignment
belong to the dataset, not to this network.  IDO-GS changes the training loss;
the backbone and its initialisation match the existing Head40-533 baseline.

This module imports no project code and can be copied with the new experiment
without any of the earlier model implementations.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock3D(nn.Module):
    """Two 3 x 3 x 3 convolutions with a residual connection."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        features = self.relu(self.bn1(self.conv1(x)))
        features = self.bn2(self.conv2(features))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(features + residual)


class ResNet18Head40Stem533IDOGS(nn.Module):
    """Head40-533 backbone; IDO-GS introduces no additional model parameters."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(2, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        shortcut = None
        if stride != 1 or self.in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

        stage = [
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=shortcut,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            stage.append(BasicBlock3D(self.in_channels, out_channels))
        return nn.Sequential(*stage)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final convolutional feature volume before global pooling."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, start_dim=1)
        return self.fc(self.dropout(features))


def resnet18_3d(
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> ResNet18Head40Stem533IDOGS:
    """Construct the standalone model; the two-class default has 33,141,954 parameters."""
    return ResNet18Head40Stem533IDOGS(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def architecture_metadata() -> dict[str, object]:
    """Describe the default network and the dataset conventions it expects."""
    return {
        "model": "ResNet18Head40Stem533IDOGS",
        "experiment": "IDO-GS",
        "block": "BasicBlock3D",
        "layers": [2, 2, 2, 2],
        "channels": [64, 128, 256, 512],
        "input_channels": 1,
        "num_classes": 2,
        "dropout_rate": 0.5,
        "trainable_parameters_default": 33_141_954,
        "input_depth": 40,
        "supported_cross_sections": [[29, 29], [37, 37], [49, 49]],
        "stem_conv_kernel": [5, 3, 3],
        "stem_conv_stride": [1, 1, 1],
        "stem_conv_padding": [2, 1, 1],
        "stem_pool_kernel": [3, 3, 3],
        "stem_pool_stride": [1, 1, 1],
        "stem_pool_padding": [1, 1, 1],
        "residual_stage_strides": [1, 2, 2, 2],
        "batch_norm_eps": 1e-5,
        "batch_norm_momentum": 0.1,
        "classifier_pool": "AdaptiveAvgPool3d(1,1,1)",
        "direction_standardized": True,
        "standardized_head_side": "high_depth_index",
        "runtime_direction_rule": "bHeadUp_true_flip_D",
        "toml_crop_rule": {
            "sequence_mapping": "toml_sequence=file_sequence-1",
            "bHeadUp_false": "left15_right25",
            "bHeadUp_true": "left25_right15",
        },
    }


if __name__ == "__main__":
    network = resnet18_3d().eval()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    assert parameter_count == architecture_metadata()["trainable_parameters_default"]
    print(f"Parameters: {parameter_count:,}")
    with torch.inference_mode():
        sample = torch.zeros(1, 1, 40, 29, 29)
        logits = network(sample)
    assert logits.shape == (1, 2)
    print(f"Input {tuple(sample.shape)} -> logits {tuple(logits.shape)}")
