from __future__ import annotations

import torch
import torch.nn as nn


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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet18_3D(nn.Module):
    """Original project 3D ResNet18 baseline with adaptive global pooling."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes

        # Preserve H/W resolution in the stem, matching the original baseline.
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(7, 3, 3),
            stride=(2, 1, 1),
            padding=(3, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(2, 1, 1),
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
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def resnet18_3d(num_classes: int = 2, dropout_rate: float = 0.5) -> ResNet18_3D:
    return ResNet18_3D(num_classes=num_classes, dropout_rate=dropout_rate)


def architecture_metadata() -> dict[str, object]:
    return {
        "model": "ResNet18_3D",
        "block": "BasicBlock3D",
        "layers": [2, 2, 2, 2],
        "channels": [64, 128, 256, 512],
        "input_channels": 1,
        "num_classes": 2,
        "dropout_rate": 0.5,
        "stem_conv_kernel": [7, 3, 3],
        "stem_conv_stride": [2, 1, 1],
        "stem_pool_stride": [2, 1, 1],
        "classifier_pool": "AdaptiveAvgPool3d(1,1,1)",
    }


if __name__ == "__main__":
    model = resnet18_3d()
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for shape in ((296, 37, 37), (237, 29, 29), (361, 49, 49)):
        sample = torch.randn(1, 1, *shape)
        with torch.no_grad():
            features = model.forward_features(sample)
            output = model(sample)
        print(f"input={shape}, layer4={tuple(features.shape[2:])}, output={tuple(output.shape)}")
