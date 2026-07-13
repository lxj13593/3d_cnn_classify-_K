import torch
import torch.nn as nn


class ConvBNReLU3d(nn.Sequential):
    """3D 卷积 + 批归一化 + ReLU6 激活函数"""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        # 兼容整数或元组类型的 kernel_size
        padding = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else tuple((k - 1) // 2 for k in kernel_size)
        super(ConvBNReLU3d, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual3d(nn.Module):
    """3D 倒残差模块 (MobileNet-V2 的核心组件)"""

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual3d, self).__init__()
        self.stride = stride

        # 兼容处理元组形式的 stride 判断
        stride_is_1 = stride == 1 or stride == (1, 1, 1)

        hidden_dim = int(round(inp * expand_ratio))
        # 只有步长为 1 且输入输出通道数相同时，才使用短接残差边
        self.use_res_connect = stride_is_1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1. 升维 (1x1x1 点卷积)
            layers.append(ConvBNReLU3d(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # 2. 深度可分离卷积 (3x3x3 组卷积，组数等于输入通道数)
            ConvBNReLU3d(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 3. 降维 (1x1x1 点卷积，注意这里最后没有 ReLU)
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_3D(nn.Module):
    """
    专门为 (296, 37, 37) 异向长条体数据定制的 3D MobileNet-V2
    完美对齐了自定义 ResNet-18 的 10x5x5 最终感受野。
    """

    def __init__(self, num_classes=2, input_channels=1, width_mult=1.0, dropout_rate=0.2):
        super(MobileNetV2_3D, self).__init__()
        block = InvertedResidual3d
        input_channel = 32
        last_channel = 1280

        # 🔥 终极定制区：完全对齐 ResNet-18 的空间特征下采样节奏
        # 配置表参数：t(升维倍数), c(输出通道), n(重复次数), s(Z, X, Y 步长)
        interverted_residual_setting = [
            [1, 16, 1, (1, 1, 1)],  # 尺寸保持: 148 x 37 x 37
            [6, 24, 2, (2, 2, 2)],  # 尺寸减半:  74 x 19 x 19
            [6, 32, 3, (2, 2, 2)],  # 尺寸减半:  37 x 10 x 10
            [6, 64, 4, (2, 2, 2)],  # 尺寸减半:  19 x  5 x  5 (🔥 XY 到达 5x5！)
            [6, 96, 3, (1, 1, 1)],  # 尺寸保持:  19 x  5 x  5
            [6, 160, 3, (2, 1, 1)],  # 🔥 魔法：仅 Z 轴减半，XY 锁死！尺寸变: 10 x 5 x 5
            [6, 320, 1, (1, 1, 1)],  # 尺寸保持:  10 x  5 x  5
        ]

        # 1. 第一层标准卷积 (初始降采样，使用 (2,1,1) 起手式保护 XY 轴)
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # Stem 层对齐你的 ResNet 保护策略
        features = [ConvBNReLU3d(input_channels, input_channel, stride=(2, 1, 1))]

        # 2. 依次构建倒残差模块
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 3. 构建最后特征提取层 (1x1x1 卷积升维)
        features.append(ConvBNReLU3d(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # 4. 全局平均池化与分类器
        # 使用 AdaptiveAvgPool3d 固定输出大小为 1x1x1，非常稳健
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        # 初始化模型权重
        self._initialize_weights()

    def forward(self, x):
        # 1. 提取特征图
        x = self.features(x)

        # 2. 全局池化 (把 10x5x5 压平成一维)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 3. 全连接层分类
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ================= 测试代码 =================
if __name__ == '__main__':
    # 你可以单独运行这个文件来测试网络是否报错，并查看参数量
    model = MobileNetV2_3D(num_classes=2, input_channels=1, dropout_rate=0.5)

    # 模拟一个 Batch=2, 通道=1, Z=296, Y=37, X=37 的输入数据
    dummy_input = torch.randn(2, 1, 296, 37, 37)

    # 前向传播测试
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")