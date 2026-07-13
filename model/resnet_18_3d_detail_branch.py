import torch
import torch.nn as nn

# =====================================================================
# 原始 ResNet18-3D 主分支 + Layer1 高分辨率细节分支
#
# 主分支：标准 ResNet18-3D backbone，使用 Layer4 输出进行全局平均池化，得到 512 维语义特征。
# 细节分支：从 Layer1 引出高分辨率浅层特征，用 1x1x1 Conv 64->128，
#           再用 AvgPool + MaxPool 提取全局细节响应。
# 最终特征：主分支 512 + 细节分支 128(avg) + 128(max) = 768。
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
    def __init__(self, num_classes=2, dropout_rate=0.5, detail_planes=128):
        self.inplanes = 64
        super(ResNet18_3D, self).__init__()

        self.num_classes = num_classes
        self.detail_planes = detail_planes

        # 修改 conv1，不对高宽进行下采样，保护 37x37 平面分辨率
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

        # --- Layer1 高分辨率细节分支 ---
        # 只把 Layer1 作为全局细节补充，不直接拼入大特征图，减少浅层噪声干扰。
        self.detail_conv = nn.Conv3d(64, self.detail_planes, kernel_size=1, bias=False)
        self.detail_bn = nn.BatchNorm3d(self.detail_planes)
        self.detail_relu = nn.ReLU(inplace=True)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)

        # 主分支 512 + 细节分支 AvgPool 128 + MaxPool 128 = 768
        self.main_planes = 512
        self.fc_in_planes = self.main_planes + self.detail_planes * 2
        self.fc = nn.Linear(self.fc_in_planes, num_classes)

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

        # Layer1 高分辨率特征，保留给细节分支
        x1 = self.layer1(x)  # [B, 64, 74, 37, 37]

        # ---------------- 原始 ResNet18 主分支 ----------------
        x2 = self.layer2(x1)  # [B, 128, 37, 19, 19]
        x3 = self.layer3(x2)  # [B, 256, 19, 10, 10]
        x4 = self.layer4(x3)  # [B, 512, 10, 5, 5]

        main_feat = self.avgpool(x4)       # [B, 512, 1, 1, 1]
        main_feat = torch.flatten(main_feat, 1)  # [B, 512]

        # ---------------- Layer1 细节分支 ----------------
        detail_feat = self.detail_conv(x1)
        detail_feat = self.detail_bn(detail_feat)
        detail_feat = self.detail_relu(detail_feat)  # [B, 128, 74, 37, 37]

        detail_avg = self.avgpool(detail_feat)         # [B, 128, 1, 1, 1]
        detail_max = self.maxpool_global(detail_feat)  # [B, 128, 1, 1, 1]
        detail_avg = torch.flatten(detail_avg, 1)      # [B, 128]
        detail_max = torch.flatten(detail_max, 1)      # [B, 128]

        # 主分支语义特征 + 浅层细节平均响应 + 浅层细节最大响应
        x = torch.cat([main_feat, detail_avg, detail_max], dim=1)  # [B, 768]
        x = self.dropout(x)
        x = self.fc(x)  # [B, num_classes]

        return x


def resnet18_3d(num_classes=2):
    return ResNet18_3D(num_classes=num_classes)


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 296, 37, 37)
    model = resnet18_3d(num_classes=2)

    print(f"输入尺寸: {input_tensor.shape}")
    with torch.no_grad():
        output = model(input_tensor)

    print(f"输出尺寸: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
