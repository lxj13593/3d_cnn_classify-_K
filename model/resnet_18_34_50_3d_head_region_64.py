import torch
import torch.nn as nn


# 1. 定义标准的 BasicBlock 模块（用于 ResNet-18/34）
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

        # 如果维度不匹配，通过 downsample 调整残差
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 2. 定义 Bottleneck 模块（用于 ResNet-50/101/152）
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


# 2. 定义 ResNet 主干网络
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        
        # 保存 num_classes 供初始化函数使用
        self.num_classes = num_classes

        # --- 标准 ResNet 的第一层配置 ---
        # # 输入假设为 (Batch, 3, 16, 112, 112) 或类似尺寸
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU(inplace=True)
        #
        # # 标准最大池化：在时间、高、宽三个维度都进行 2倍下采样
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Head-region input depth is 64, so keep depth resolution in the stem.
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # Keep depth resolution in the pooling layer too.
        # Shape schedule for [B, 1, 64, 37, 37]:
        # stem/layer1: 64x37x37 -> layer2: 32x19x19
        # -> layer3: 16x10x10 -> layer4: 8x5x5.
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # --- 4个残差层 ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出固定为 1x1x1
        self.dropout = nn.Dropout(dropout_rate)  # Dropout正则化
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # ResNet-50 输出 2048 维

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
        x = self.maxpool(x)  # stride=1，stem阶段不下采样

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)

        return x
    # def forward(self, x):
    #     print(f"【输入数据】尺寸: {x.shape}")
    #
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     print(f"【conv1】后尺寸:   {x.shape}")
    #
    #     x = self.maxpool(x)
    #     print(f"【maxpool】后尺寸: {x.shape}")
    #
    #     x = self.layer1(x)
    #     print(f"【layer1】后尺寸:  {x.shape}")
    #
    #     x = self.layer2(x)
    #     print(f"【layer2】后尺寸:  {x.shape}")
    #
    #     x = self.layer3(x)
    #     print(f"【layer3】后尺寸:  {x.shape}")
    #
    #     x = self.layer4(x)
    #     print(f"【layer4】后尺寸:  {x.shape}")
    #
    #     x = self.avgpool(x)
    #     print(f"【avgpool】后尺寸: {x.shape}")
    #
    #     x = torch.flatten(x, 1)
    #     print(f"【flatten】后尺寸: {x.shape}")
    #
    #     x = self.dropout(x)
    #     x = self.fc(x)
    #     print(f"【最终输出】尺寸: {x.shape}\n" + "-" * 40)
    #
    #     return x


# 3. 实例化 ResNet-18
def resnet18_3d(num_classes=2):
    """Constructs a ResNet-18 3D model_best_last."""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet34_3d(num_classes=2):
    """Constructs a ResNet-34 3D model_best_last."""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


# 4. 实例化 ResNet-50
def resnet50_3d(num_classes=2):
    """Constructs a ResNet-50 3D model_best_last."""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟头部 ROI 输入：[Batch, Channel, Depth, Height, Width]
    input_tensor = torch.randn(1, 1, 64, 37, 37)
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"输出尺寸: {output.shape}")  # 应为 [1, 2]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
