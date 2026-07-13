import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 核心逻辑：Layer2(128)、Layer3(256) 和 Layer4(512) 特征图拼接成 896 维，
# 后降维为 512 维。空间尺寸统一对齐到 Layer2 的分辨率。
#
# 本版本额外修改：
# 将 BasicBlock 中的第二个 3x3x3 卷积替换为非对称 3D 卷积：
# 3x1x1 -> 1x3x1 -> 1x1x3。
# conv1 保持原始 3x3x3 卷积，避免影响 stride 下采样和通道变化。
# =====================================================================


class AsymmetricConv3D(nn.Module):
    """
    非对称 3D 卷积模块。
    用三个一维方向卷积近似原来的 3x3x3 卷积：
        3x1x1 -> 1x3x1 -> 1x1x3
    输入输出通道数保持一致，空间尺寸保持不变。
    """
    def __init__(self, channels):
        super(AsymmetricConv3D, self).__init__()
        self.conv_d = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            bias=False
        )
        self.conv_h = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 3, 1),
            stride=1,
            padding=(0, 1, 0),
            bias=False
        )
        self.conv_w = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, 3),
            stride=1,
            padding=(0, 0, 1),
            bias=False
        )

    def forward(self, x):
        x = self.conv_d(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # conv1 保持原始 3x3x3，用于通道变化和可能的下采样
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # conv2 改为非对称 3D 卷积，输出尺寸和通道数与原 conv2 一致
        self.conv2 = AsymmetricConv3D(planes)
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


class ResNet18_3D(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        self.inplanes = 64
        super(ResNet18_3D, self).__init__()

        self.num_classes = num_classes

        # 修改 conv1，不对高宽进行下采样
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # 修改 maxpool，同样保护高宽方向的分辨率
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))

        # --- 4个残差层 ---
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # --- 多尺度融合：Layer2 + Layer3 + Layer4 ---
        self.combined_planes = 128 + 256 + 512

        # --- 1x1x1 卷积降维：896 -> 512 ---
        self.reduce_planes = 512
        self.reduce_conv = nn.Conv3d(self.combined_planes, self.reduce_planes, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm3d(self.reduce_planes)
        self.reduce_relu = nn.ReLU(inplace=True)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.reduce_planes, num_classes)

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
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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

        # --- 逐步提取 Layer2, Layer3, Layer4 的特征 ---
        x2 = self.layer2(x)   # [B, 128, D2, H2, W2]
        x3 = self.layer3(x2)  # [B, 256, D3, H3, W3]
        x4 = self.layer4(x3)  # [B, 512, D4, H4, W4]

        # --- 将 Layer3 和 Layer4 的空间尺寸统一到 Layer2 ---
        target_size = x2.shape[2:]
        x3_upsampled = F.interpolate(x3, size=target_size, mode='trilinear', align_corners=False)
        x4_upsampled = F.interpolate(x4, size=target_size, mode='trilinear', align_corners=False)

        # --- 拼接三个尺度特征：[B, 896, D2, H2, W2] ---
        fused_feat = torch.cat([x2, x3_upsampled, x4_upsampled], dim=1)

        # --- 1x1x1 卷积降维：[B, 512, D2, H2, W2] ---
        fused_feat = self.reduce_conv(fused_feat)
        fused_feat = self.reduce_bn(fused_feat)
        fused_feat = self.reduce_relu(fused_feat)

        # --- 分类头 ---
        x = self.avgpool(fused_feat)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18_3d(num_classes=2):
    return ResNet18_3D(num_classes=num_classes)


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)

    print(f"输出尺寸: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
