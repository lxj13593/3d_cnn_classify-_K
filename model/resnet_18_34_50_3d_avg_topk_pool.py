import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. BasicBlock for ResNet-18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 2. Bottleneck for ResNet-50/101/152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AvgTopKPool3D(nn.Module):
    """
    AvgPool + Top-k Pooling for small/local defect classification.

    Input:  x [B, C, D, H, W]
    Output: pooled feature [B, 2C]

    - avg branch: global average over all spatial positions, stable global representation
    - top-k branch: channel-wise mean of the top-k spatial responses, keeps sparse strong local responses
    """
    def __init__(self, topk_ratio=0.10):
        super(AvgTopKPool3D, self).__init__()
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1].")
        self.topk_ratio = topk_ratio

    def forward(self, x):
        # Global average branch: [B, C]
        avg_feat = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).flatten(1)

        # Top-k branch: [B, C, D*H*W] -> [B, C]
        b, c, d, h, w = x.shape
        x_flat = x.flatten(2)
        num_positions = x_flat.size(-1)
        k = max(1, int(math.ceil(num_positions * self.topk_ratio)))

        topk_values = torch.topk(x_flat, k=k, dim=2, largest=True, sorted=False).values
        topk_feat = topk_values.mean(dim=2)

        # Concatenate stable global feature and sparse local response feature
        pooled = torch.cat([avg_feat, topk_feat], dim=1)
        return pooled


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5, topk_ratio=0.10):
        self.inplanes = 64
        super(ResNet3D, self).__init__()

        self.num_classes = num_classes
        self.topk_ratio = topk_ratio

        # Keep original ResNet18 front-end settings for fair comparison.
        # Input: [B, 1, 296, 37, 37]
        # conv1: D 296 -> 148, H/W keep 37x37
        self.conv1 = nn.Conv3d(
            1, 64,
            kernel_size=(7, 3, 3),
            stride=(2, 1, 1),
            padding=(3, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # maxpool: D 148 -> 74, H/W keep 37x37
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Replace single GAP with AvgPool + channel-wise Top-k Pooling.
        # For ResNet18, layer4 output is about [B, 512, 10, 5, 5].
        # topk_ratio=0.10 means roughly top 25 positions per channel.
        self.pool = AvgTopKPool3D(topk_ratio=topk_ratio)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Factory functions
def resnet18_3d(num_classes=2, dropout_rate=0.5, topk_ratio=0.10):
    return ResNet3D(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        topk_ratio=topk_ratio
    )


def resnet34_3d(num_classes=2, dropout_rate=0.5, topk_ratio=0.10):
    return ResNet3D(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        topk_ratio=topk_ratio
    )


def resnet50_3d(num_classes=2, dropout_rate=0.5, topk_ratio=0.10):
    return ResNet3D(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        topk_ratio=topk_ratio
    )


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    model = resnet18_3d(num_classes=2, topk_ratio=0.10)

    print(f"Input shape: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
