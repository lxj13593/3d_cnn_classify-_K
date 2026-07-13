import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 核心逻辑：Layer2(128)、Layer3(256) 和 Layer4(512) 特征图拼接成 896 维
# 后，降维为 512 维。空间尺寸统一对齐到 Layer2 的分辨率。
# =====================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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

        # --- 修改点 1：动态计算拼接后的总通道数 (128 + 256 + 512 = 896) ---
        self.combined_planes = 128 + 256 + 512

        # --- 修改点 2：1x1x1 卷积的输入通道数变更为 896 ---
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
        
        # --- 修改点 3：逐步提取 Layer2, Layer3, Layer4 的特征 ---
        x2 = self.layer2(x)   # 尺寸: [B, 128, 37, 19, 19]
        x3 = self.layer3(x2)  # 尺寸: [B, 256, 19, 10, 10]
        x4 = self.layer4(x3)  # 尺寸: [B, 512, 10, 5, 5]

        # --- 修改点 4：将 Layer3 和 Layer4 的空间尺寸三线性上采样到 Layer2 的尺寸 (37x19x19) ---
        target_size = x2.shape[2:]
        x3_upsampled = F.interpolate(x3, size=target_size, mode='trilinear', align_corners=False)
        x4_upsampled = F.interpolate(x4, size=target_size, mode='trilinear', align_corners=False)

        # --- 修改点 5：将三个尺度的特征在通道维度拼接 -> 尺寸: [B, 896, 37, 19, 19] ---
        fused_feat = torch.cat([x2, x3_upsampled, x4_upsampled], dim=1)
        
        # 1x1x1 卷积降维 -> 尺寸: [B, 512, 37, 19, 19]
        fused_feat = self.reduce_conv(fused_feat)
        fused_feat = self.reduce_bn(fused_feat)
        fused_feat = self.reduce_relu(fused_feat)
        
        # 分类头处理
        x = self.avgpool(fused_feat)  # 变为 [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)       # 变为 [B, 512]
        x = self.dropout(x)
        x = self.fc(x)                # 最终分类输出 [B, num_classes]

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