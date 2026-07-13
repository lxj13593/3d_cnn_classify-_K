import torch
import torch.nn as nn
import torch.nn.functional as F  # 引入 functional 用于三线性插值上采样

# =====================================================================
# 核心逻辑：Layer3(256) 和 Layer4(512) 特征图拼接成 768 维后，降维为 512 维
# =====================================================================

# 1. 定义标准的 BasicBlock 模块（用于 ResNet-18）
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


# 2. 定义 ResNet-18 3D 主干网络
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

        # --- 4个残差层（ResNet-18 的标准层数配置 [2, 2, 2, 2]） ---
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 动态计算拼接后的通道数 (Layer3的256 + Layer4的512 = 768通道)
        self.combined_planes = 256 + 512

        # --- 修正点：根据您的要求，1x1x1 卷积直接将通道数从 768 降维至 512 ---
        self.reduce_planes = 512
        self.reduce_conv = nn.Conv3d(self.combined_planes, self.reduce_planes, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm3d(self.reduce_planes)
        self.reduce_relu = nn.ReLU(inplace=True)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出固定为 1x1x1
        self.dropout = nn.Dropout(dropout_rate)         # Dropout
        
        # 分类线性层的输入特征同步更新为降维后的 512 维
        self.fc = nn.Linear(self.reduce_planes, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化 - 标准 ResNet 初始化策略"""
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
        x = self.layer2(x)
        
        # 1. 提取多尺度特征
        x3 = self.layer3(x)  # 尺寸: [B, 256, 19, 10, 10]
        x4 = self.layer4(x3) # 尺寸: [B, 512, 10, 5, 5]

        # 2. 将 Layer4 特征三线性上采样到 Layer3 的空间尺寸 (19x10x10)
        target_size = x3.shape[2:]
        x4_upsampled = F.interpolate(x4, size=target_size, mode='trilinear', align_corners=False)

        # 3. 沿通道维度 (dim=1) 拼接 -> 尺寸变为: [B, 768, 19, 10, 10]
        fused_feat = torch.cat([x3, x4_upsampled], dim=1)
        
        # 4. 通过 1x1x1 卷积完成 768 -> 512 的降维 -> 尺寸变为: [B, 512, 19, 10, 10]
        fused_feat = self.reduce_conv(fused_feat)
        fused_feat = self.reduce_bn(fused_feat)
        fused_feat = self.reduce_relu(fused_feat)
        
        # 5. 分类头处理
        x = self.avgpool(fused_feat)  # 变为 [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)       # 变为 [B, 512]
        x = self.dropout(x)
        x = self.fc(x)                # 最终分类输出 [B, num_classes]

        return x


# 3. 简化后的 ResNet-18 专属实例化入口
def resnet18_3d(num_classes=2):
    """构建专属于 ResNet-18 的 3D 模型框架"""
    model = ResNet18_3D(num_classes=num_classes)
    return model


# --- 测试验证代码 ---
if __name__ == "__main__":
    # 模拟输入：Batch=1, 通道=1, 帧数=296, 高=37, 宽=37
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    
    # 实例化精简后的模型
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    output = model(input_tensor)
    
    print(f"输出尺寸: {output.shape}")  # 稳定输出 [1, 2]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")