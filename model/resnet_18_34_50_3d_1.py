import torch
import torch.nn as nn

#修改特征图尺寸19*10*10

# 1. 定义标准的 BasicBlock 模块（用于 ResNet-18/34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 解耦 stride，如果是整数则转为元组
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

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

        self.num_classes = num_classes

        # 【调整点 1】改变 conv1 的步长和 padding
        # 输入: (1, 1, 296, 37, 37) -> 输出: (1, 64, 148, 19, 19)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 3, 3), stride=(2, 2, 2), padding=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # 【调整点 2】改变 maxpool 步长，高宽方向不动
        # 输入: (1, 64, 148, 19, 19) -> 输出: (1, 64, 74, 19, 19)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))

        # --- 4个残差层精确控制尺寸 ---
        # layer1: stride=1 -> 输出: (1, 64, 74, 19, 19)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)

        # layer2: 只在深度方向下采样 -> 输出: (1, 128, 37, 19, 19)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1, 1))

        # layer3: 深度和高宽同时下采样 -> 输出: (1, 256, 19, 9, 9)
        # 37/2 = 18.5 -> 19 ; 19/2 = 9.5 -> 9 (PyTorch 卷积向下取整，配合 padding=1 刚好是 19 和 9)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))

        # layer4: 已经达到目标尺寸 19x9x9，闭合下采样，只提升通道数 -> 输出: (1, 512, 19, 9, 9)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

        # 转换为元组以便进行比较
        if isinstance(stride, int):
            stride_tuple = (stride, stride, stride)
        else:
            stride_tuple = stride

        # 如果任何一个维度的 stride != 1 或者通道不匹配，则进行 downsample
        if any(s != 1 for s in stride_tuple) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride_tuple, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"【输入数据】尺寸: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(f"【conv1】后尺寸:   {x.shape}")

        x = self.maxpool(x)
        # print(f"【maxpool】后尺寸: {x.shape}")

        x = self.layer1(x)
        # print(f"【layer1】后尺寸:  {x.shape}")

        x = self.layer2(x)
        # print(f"【layer2】后尺寸:  {x.shape}")

        x = self.layer3(x)
        # print(f"【layer3】后尺寸:  {x.shape}")

        x = self.layer4(x)
        # print(f"【layer4】后尺寸:  {x.shape}")

        x = self.avgpool(x)
        # print(f"【avgpool】后尺寸: {x.shape}")

        x = torch.flatten(x, 1)
        # print(f"【flatten】后尺寸: {x.shape}")

        x = self.dropout(x)
        x = self.fc(x)
        # print(f"【最终输出】尺寸: {x.shape}\n" + "-" * 40)

        return x


def resnet18_3d(num_classes=2):
    return ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# # --- 测试代码 ---
# if __name__ == "__main__":
#     input_tensor = torch.randn(1, 1, 296, 37, 37)
#     model = resnet18_3d(num_classes=2)
#
#     output = model(input_tensor)