import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    """3D 挤压和激励 (Squeeze-and-Excitation) 模块"""

    def __init__(self, channels, reduced_dim):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.SiLU(),  # EfficientNet 官方指定使用 SiLU (Swish)
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze 阶段
        w = self.avg_pool(x).view(b, c)
        # Excitation 阶段
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class MBConv3D(nn.Module):
    """3D 移动倒残差块 (Mobile Inverted Bottleneck Convolution)"""

    def __init__(self, inp, oup, stride, expand_ratio, kernel_size, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConv3D, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        # 只有在步长为1 且 输入输出通道完全一致时才启用残差连接
        self.use_res_connect = (inp == oup) and (stride == 1 or stride == (1, 1, 1))
        self.drop_connect_rate = drop_connect_rate

        hidden_dim = int(inp * expand_ratio)
        padding = (kernel_size - 1) // 2

        layers = []
        # 1. 升维阶段 (1x1x1 点卷积)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU()
            ])

        # 2. 深度可分离卷积阶段 (Depthwise Conv 3D)
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU()
        ])

        # 3. SE 注意力机制模块
        if se_ratio:
            reduced_dim = max(1, int(inp * se_ratio))
            layers.append(SEBlock3D(hidden_dim, reduced_dim))

        # 4. 降维线性阶段 (Pointwise Conv 3D, 注意末尾无激活函数)
        layers.extend([
            nn.Conv3d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm3d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            if self.training and self.drop_connect_rate > 0:
                # 实现 Stochastic Depth (DropConnect)
                keep_prob = 1.0 - self.drop_connect_rate
                batch_size = x.size(0)
                random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1, 1], dtype=x.dtype, device=x.device)
                binary_tensor = torch.floor(random_tensor)
                return x + (self.conv(x) / keep_prob) * binary_tensor
            else:
                return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNetB0_3D(nn.Module):
    """
    针对 3D 异向体数据定制的 EfficientNet-B0。
    空间维度的下采样节奏与你之前设计的 ResNet-18 和 MobileNet-V2 保持完美同步。
    """

    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.2, drop_connect_rate=0.2):
        super(EfficientNetB0_3D, self).__init__()
        self.num_classes = num_classes

        # 🔥 核心节奏控制区
        # 每一行的参数分别对应：t(升维倍数), c(输出通道数), n(当前组重复次数), s(Z, X, Y轴步长), k(核大小)
        efficientnet_b0_setting = [
            [1, 16, 1, (1, 1, 1), 3],  # Stage 1 -> 尺寸保持: 148 x 37 x 37
            [6, 24, 2, (2, 2, 2), 3],  # Stage 2 -> 尺寸减半:  74 x 19 x 19
            [6, 40, 2, (2, 2, 2), 5],  # Stage 3 -> 尺寸减半:  37 x 10 x 10
            [6, 80, 3, (2, 2, 2), 3],  # Stage 4 -> 尺寸减半:  19 x  5 x  5  (XY轴到达边界)
            [6, 112, 3, (1, 1, 1), 5],  # Stage 5 -> 尺寸保持:  19 x  5 x  5
            [6, 192, 4, (2, 1, 1), 5],  # Stage 6 -> 仅Z轴减半:  10 x  5 x  5
            [6, 320, 1, (1, 1, 1), 3],  # Stage 7 -> 尺寸保持:  10 x  5 x  5
        ]

        # 1. Stem 头部层：使用 (2, 1, 1) 步长起手，保护高宽 (X, Y) 方向的分辨率
        in_channels = 32
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, in_channels, kernel_size=3, stride=(2, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.SiLU()
        )

        # 2. 纵向堆叠所有的 MBConv3D 块
        layers = []
        for t, c, n, s, k in efficientnet_b0_setting:
            for i in range(n):
                # 仅在每组的第一个Block应用设定步长，后续重复的Block步长回归(1,1,1)
                stride = s if i == 0 else (1, 1, 1)
                layers.append(
                    MBConv3D(inp=in_channels, oup=c, stride=stride, expand_ratio=t, kernel_size=k,
                             drop_connect_rate=drop_connect_rate)
                )
                in_channels = c

        self.blocks = nn.Sequential(*layers)

        # 3. Head 卷积放大层
        self.last_channels = 1280
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, self.last_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.last_channels),
            nn.SiLU()
        )

        # 4. 全局池化与最终的线性分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.last_channels, num_classes)
        )

        # 全网络初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ================= 结构验证与尺寸测试 =================
if __name__ == '__main__':
    # 模拟你的异向长条输入数据：Batch=1, Channel=1, Z=296, X=37, Y=37
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    model = EfficientNetB0_3D(num_classes=2, input_channels=1)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"输出分类张量尺寸: {output.shape}")  # 应平稳输出 [1, 2]
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")