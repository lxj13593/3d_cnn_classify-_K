import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer3D, self).__init__()
        # 1x1x1 瓶颈层 (Bottleneck)，用于降维和减少计算量
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        # 3x3x3 卷积层，提取 3D 空间特征
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock3D(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock3D, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer3D(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        # 🔥 🔥 核心修改 1：添加 ceil_mode=True。
        # 配合标准的等向 stride=2，能够完美把 37->19->10->5 的奇数边界对齐
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True))


class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=1000, in_channels=3):
        super(DenseNet3D, self).__init__()

        # 1. 初始前级特征提取 (Stem 结构)
        self.features = nn.Sequential()

        # 🔥 🔥 核心修改 2：异向化 Stem 层，完全复刻你 ResNet-18 的保护高宽分辨率策略
        self.features.add_module('conv0', nn.Conv3d(in_channels, num_init_features,
                                                    kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1),
                                                    bias=False))
        self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        # 同样在 maxpool 处锁定高宽方向，仅对 Z 轴进行二倍下采样
        self.features.add_module('pool0', nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)))

        # 2. 循环构建四个 DenseBlock 和对应的 Transition
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition3D(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 3. 最后的全局归一化
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # 4. 分类器分类层
        self.classifier = nn.Linear(num_features, num_classes)

        # 5. 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        # 全局 3D 平均池化，将 (B, C, 10, 5, 5) 压平成 (B, C, 1, 1, 1)
        out = F.adaptive_avg_pool3d(features, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121_3d(num_classes=1000, in_channels=3, **kwargs):
    """构建统一节奏后的 3D DenseNet-121 模型"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, num_classes=num_classes, in_channels=in_channels, **kwargs)


# --- 尺寸验证脚本 ---
if __name__ == "__main__":
    # 使用你统一的长条异向体输入：Batch=1, Channel=1, Z=296, Y=37, X=37
    model = densenet121_3d(num_classes=2, in_channels=1)

    dummy_input = torch.randn(1, 1, 296, 37, 37)

    # 打印前向传播时的特征图变换过程
    print(f"输入数据形状: {dummy_input.shape}")

    # 逐步追踪空间特征图变化
    x = model.features.conv0(dummy_input)
    x = model.features.norm0(x)
    x = model.features.relu0(x)
    print(f"经过 conv0 后的形状: {x.shape}")

    x = model.features.pool0(x)
    print(f"经过 pool0 (Stem 结束) 后的形状: {x.shape}")

    x = model.features.denseblock1(x)
    x = model.features.transition1(x)
    print(f"经过 Transition 1 后的形状: {x.shape}")

    x = model.features.denseblock2(x)
    x = model.features.transition2(x)
    print(f"经过 Transition 2 后的形状: {x.shape}")

    x = model.features.denseblock3(x)
    x = model.features.transition3(x)
    print(f"经过 Transition 3 (即最后进入高阶 Block4 前) 的特征图形状: {x.shape}")

    x = model.features.denseblock4(x)
    x = model.features.norm5(x)
    final_features = model.features.relu5(x)
    print(f"\n🔥 进入全局平均池化前，最终特征图的完整形状 (C, D, H, W): {final_features.shape[1:]}")

    output = model(dummy_input)
    print(f"模型最终输出分类张量形状: {output.shape}")