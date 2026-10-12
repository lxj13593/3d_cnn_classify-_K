from __future__ import annotations

import torch
import torch.nn as nn


class ELA3D(nn.Module):
    """ELA-S extended to D/H/W, with a shared Conv1d + GroupNorm encoder.

    Each axis retains its coordinates while averaging over the other two
    spatial axes. The three sigmoid maps are broadcast and multiplied into
    the original feature tensor. This is not foreground segmentation.
    """

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 5,
        norm_groups: int = 16,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 8:
            raise ValueError("ELA-S channels must be a positive multiple of 8")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ELA kernel_size must be positive and odd")
        if norm_groups <= 0 or channels % norm_groups:
            raise ValueError("ELA norm_groups must divide channels")
        self.channels = channels
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels // 8,
            bias=False,
        )
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(
                f"ELA3D expects [B,{self.channels},D,H,W], got {tuple(x.shape)}"
            )
        batch, channels, depth, height, width = x.shape
        depth_weights = self.sigmoid(self.norm(self.conv(x.mean(dim=(3, 4)))))
        height_weights = self.sigmoid(self.norm(self.conv(x.mean(dim=(2, 4)))))
        width_weights = self.sigmoid(self.norm(self.conv(x.mean(dim=(2, 3)))))
        return (
            x
            * depth_weights.reshape(batch, channels, depth, 1, 1)
            * height_weights.reshape(batch, channels, 1, height, 1)
            * width_weights.reshape(batch, channels, 1, 1, width)
        )


class BasicBlock3D(nn.Module):
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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Head40Stem733ELALayer2(nn.Module):
    """Original direction-unified Head40-733, with one ELA after Layer2."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(7, 3, 3),
            stride=(1, 1, 1),
            padding=(3, 1, 1),
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
        # Initialize the original backbone first, preserving its initial weights
        # when constructed with the same seed as the original 733 model.
        self._initialize_weights()
        self.ela = ELA3D(channels=128, kernel_size=5, norm_groups=16)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

        layers = [
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.ela(self.layer2(x))
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def resnet18_3d(
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> ResNet18Head40Stem733ELALayer2:
    return ResNet18Head40Stem733ELALayer2(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def attention_metadata() -> dict[str, object]:
    return {
        "module": "ELA3D",
        "variant": "ELA-S_3D_extension",
        "count": 1,
        "position": "after_layer2_before_layer3",
        "channels": 128,
        "axes": ["D", "H", "W"],
        "axis_pool": "mean_over_other_two_spatial_axes",
        "conv1d_kernel_size": 5,
        "conv1d_groups": 16,
        "conv1d_bias": False,
        "groupnorm_groups": 16,
        "groupnorm_eps": 1e-5,
        "shared_axis_encoder": True,
        "activation": "Sigmoid",
        "fusion": "input_times_depth_times_height_times_width",
        "channel_reduction": False,
        "extra_residual": False,
        "weight_floor": None,
    }


def architecture_metadata() -> dict[str, object]:
    return {
        "model": "ResNet18Head40Stem733ELALayer2",
        "block": "BasicBlock3D",
        "layers": [2, 2, 2, 2],
        "channels": [64, 128, 256, 512],
        "input_channels": 1,
        "num_classes": 2,
        "dropout_rate": 0.5,
        "input_depth": 40,
        "supported_cross_sections": [[29, 29], [37, 37], [49, 49]],
        "stem_conv_kernel": [7, 3, 3],
        "stem_conv_stride": [1, 1, 1],
        "stem_pool_stride": [1, 1, 1],
        "residual_stage_strides": [1, 2, 2, 2],
        "classifier_pool": "AdaptiveAvgPool3d(1,1,1)",
        "direction_standardized": True,
        "standardized_head_side": "high_depth_index",
        "runtime_direction_rule": "bHeadUp_true_flip_D",
        "toml_crop_rule": {
            "sequence_mapping": "toml_sequence=file_sequence-1",
            "bHeadUp_false": "left15_right25",
            "bHeadUp_true": "left25_right15",
        },
        "attention": attention_metadata(),
        "runtime_hw_crop": False,
        "canny_edge_enhancement": False,
    }
