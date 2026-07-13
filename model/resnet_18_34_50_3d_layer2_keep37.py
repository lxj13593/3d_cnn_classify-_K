import torch
import torch.nn as nn


# 1. 定义标准的 BasicBlock 模块（用于 ResNet-18/34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
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


# 2. 定义 Bottleneck 模块（用于 ResNet-50/101/152）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
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


class ResNet3D(nn.Module):
    """
    ResNet3D with layer2 H/W resolution preserved.

    Compared with the original ResNet3D:
    - conv1 unchanged: stride=(2, 1, 1)
    - maxpool unchanged: stride=(2, 1, 1)
    - layer2 changed: stride=2 -> stride=(2, 1, 1)

    For input [B, 1, 296, 37, 37], approximate feature sizes are:
    original:
        conv1    [B, 64, 148, 37, 37]
        maxpool  [B, 64,  74, 37, 37]
        layer1   [B, 64,  74, 37, 37]
        layer2   [B,128,  37, 19, 19]
        layer3   [B,256,  19, 10, 10]
        layer4   [B,512,  10,  5,  5]

    this version:
        conv1    [B, 64, 148, 37, 37]
        maxpool  [B, 64,  74, 37, 37]
        layer1   [B, 64,  74, 37, 37]
        layer2   [B,128,  37, 37, 37]
        layer3   [B,256,  19, 19, 19]
        layer4   [B,512,  10, 10, 10]
    """

    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes

        # 原版 stem 不动：只在 D/长度方向下采样，不压 H/W
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

        # 原版 maxpool 不动：继续只压 D/长度方向，不压 H/W
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )

        # layer1 不下采样，保持 [74, 37, 37]
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 核心改动：layer2 只在 D/长度方向下采样，H/W 保持 37x37
        # 原版是 stride=2，会把 [37,37] 压成 [19,19]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1, 1))

        # layer3/layer4 保持原版 stride=2，让后续仍然逐步聚合语义特征
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化：保持和原版一致。"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
                    bias=False,
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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# 3. 实例化 ResNet-18

def resnet18_3d(num_classes=2):
    """Constructs a ResNet-18 3D model."""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def resnet34_3d(num_classes=2):
    """Constructs a ResNet-34 3D model."""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


# 4. 实例化 ResNet-50

def resnet50_3d(num_classes=2):
    """Constructs a ResNet-50 3D model."""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"输出尺寸: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
