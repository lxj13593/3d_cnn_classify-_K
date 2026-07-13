import torch
import torch.nn as nn

# 注意力加在每个basic_block里面，一共8次
# 参数调整，适应网络要求，ratio=8,kernel_size=3

# ==================== 3D CBAM 注意力模块 ====================
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM3D(nn.Module):
    def __init__(self, gate_channels, ratio=8, kernel_size=3):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelAttention3D(gate_channels, ratio)
        self.SpatialGate = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x_out = x * self.ChannelGate(x)
        x_out = x_out * self.SpatialGate(x_out)
        return x_out


# ==================== ResNet 核心模块 ====================

# 1. 定义集成了 3D CBAM 的 BasicBlock 模块（用于 ResNet-18/34）
class BasicBlock(nn.Module):
    expansion = 1  # ResNet-18 的 expansion 是 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 3x3x3 卷积：提取特征
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # 3x3x3 卷积：提取特征
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        # --- 实例化 3D CBAM 模块 ---
        self.cbam = CBAM3D(planes)

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

        # --- 在残差相加前，进行通道和空间注意力加权 ---
        out = self.cbam(out)

        # 如果维度不匹配，通过 downsample 调整残差
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 2. 定义标准的 Bottleneck 模块（用于 ResNet-50/101/152，未加 CBAM）
class Bottleneck(nn.Module):
    expansion = 4  # ResNet-50 的 expansion 是 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1x1 卷积：降维
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # 3x3x3 卷积：提取特征
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        # 1x1x1 卷积：升维
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
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

        # 如果维度不匹配，通过 downsample 调整残差
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 3. 定义 ResNet 主干网络
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5):
        self.inplanes = 64
        super(ResNet3D, self).__init__()

        # 保存 num_classes 供初始化函数使用
        self.num_classes = num_classes

        # 修改 conv1，不对高宽进行下采样
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # 修改 maxpool，同样保护高宽方向的分辨率
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))

        # --- 4个残差层 ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出固定为 1x1x1
        self.dropout = nn.Dropout(dropout_rate)  # Dropout正则化
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化 - 标准 ResNet 初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 卷积层：使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                # BatchNorm：weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 最后一个全连接层：使用更小的标准差
                if m.out_features == self.num_classes:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # 其他全连接层：Kaiming 初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 如果 stride != 1 或者 输入输出通道数不匹配，则需要下采样残差支路
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        # 第一个 block 负责下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # 剩余的 blocks stride 默认为 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 尺寸在这里减半

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)

        return x


# 4. 实例化 ResNet-18 (带 CBAM)
def resnet18_3d(num_classes=2):
    """Constructs a ResNet-18 3D model with CBAM."""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


# 5. 实例化 ResNet-50 (标准不带 CBAM)
def resnet50_3d(num_classes=2):
    """Constructs a ResNet-50 3D model."""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入：Batch=1, 通道=1, 帧数=16, 高=112, 宽=112
    input_tensor = torch.randn(1, 1, 16, 112, 112)

    # 更改为测试引入了 CBAM 的 resnet18_3d
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"输出尺寸: {output.shape}")  # 应为 [1, 2]
    print(f"模型参数量 (ResNet-18 + CBAM): {sum(p.numel() for p in model.parameters()):,}")