# Head40 统一方向 ResNet18 Stem333 多板五折实验说明

## 1. 实验目的

本实验用于比较 Head40 3D ResNet18 的起始卷积核设计。

本实验将 stem 卷积核设置为：

```text
3 × 3 × 3
```

对照实验使用：

```text
7 × 3 × 3
```

除 stem 卷积核及其对应 padding 外，数据、五折划分、方向统一、归一化、增强、残差结构、训练参数和检查点选择规则全部保持一致。

本实验只回答一个问题：对于深度为 40 的头部体积，局部各向同性的 `3×3×3` stem 是否优于沿 D 轴范围更大的 `7×3×3` stem。

## 2. 本实验文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem333_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem333_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem333_multiboard_3boards.py
```

本实验不修改原有模型、训练或测试文件。

按照本次实验约定，以下现有数据资产直接复用：

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py
```

不需要重新运行清单生成器。

## 3. 数据与固定划分

训练与五折验证数据：

- 22 块板
- 4,364 个样本
- normal：3,181
- defective：1,183
- 验证折样本数：872、873、874、873、872

锁定外部测试数据：

- 3 块板
- 778 个样本
- normal：556
- defective：222

本实验直接使用已经固定的逐 SampleID 五折归属，不重新分折。因此 Stem333 与 Stem733 的每一折训练样本和验证样本完全相同。

外部三板不参与模型训练、检查点选择、阈值调整或结构选择。

## 4. Head40 与方向统一

输入为根据 TOML `center2` 和 `bHeadUp` 生成的 40 层头部体积。

方向统一目标：

```text
钻头位于 D 轴高索引侧
```

运行时规则：

- `bHeadUp=false`：保持原方向。
- `bHeadUp=true`：沿 D 轴反转。

训练集中保持原方向 4,293 个，沿 D 轴反转 71 个；锁定测试集中保持原方向 769 个，沿 D 轴反转 9 个。

方向统一发生在数据加载阶段，不修改物理 RAW 文件。训练增强也不随机翻转 D 轴。

## 5. Stem333 模型结构

模型输入顺序：

```text
N × C × D × H × W
```

输入通道为 1，输入深度固定为 40，横截面支持：

```text
29 × 29
37 × 37
49 × 49
```

网络结构：

```text
Input
  → Conv3d(1→64, kernel=3×3×3, stride=1, padding=1)
  → BatchNorm3d(64)
  → ReLU
  → MaxPool3d(kernel=3×3×3, stride=1, padding=1)
  → Layer1: 2 × BasicBlock3D, 64 channels, stride 1
  → Layer2: 2 × BasicBlock3D, 128 channels, first block stride 2
  → Layer3: 2 × BasicBlock3D, 256 channels, first block stride 2
  → Layer4: 2 × BasicBlock3D, 512 channels, first block stride 2
  → AdaptiveAvgPool3d(1×1×1)
  → Flatten(512)
  → Dropout(0.5)
  → Linear(512→2)
```

每个 BasicBlock3D 包含两个 `3×3×3` 卷积。尺寸或通道变化时，残差支路使用 `1×1×1` 卷积匹配。

模型参数量：

```text
33,140,802
```

## 6. 最终特征图尺寸

Stem333 与 Stem733 的最终特征图尺寸完全相同。

原因是：

- Stem333：`kernel=(3,3,3)`、`padding=(1,1,1)`、`stride=1`
- Stem733：`kernel=(7,3,3)`、`padding=(3,1,1)`、`stride=1`

两种 stem 都保持输入尺寸不变，后续网络的 stride 也完全一致。

| 输入体积 `C×D×H×W` | Layer1 | Layer2 | Layer3 | Layer4 最终特征图 |
|---|---|---|---|---|
| `1×40×29×29` | `64×40×29×29` | `128×20×15×15` | `256×10×8×8` | `512×5×4×4` |
| `1×40×37×37` | `64×40×37×37` | `128×20×19×19` | `256×10×10×10` | `512×5×5×5` |
| `1×40×49×49` | `64×40×49×49` | `128×20×25×25` | `256×10×13×13` | `512×5×7×7` |

经过自适应全局池化后，三种输入都变为：

```text
512 × 1 × 1 × 1
```

因此该实验比较的是早期局部特征提取范围，不是特征图尺寸。

## 7. 训练设置

默认训练配置：

| 项目 | 设置 |
|---|---|
| 五折随机种子 | `42, 123, 2026, 3407, 777` |
| Epoch | 50 |
| Batch size | 4 |
| Optimizer | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-3` |
| 学习率调度 | CosineAnnealingLR |
| 最低学习率 | `1e-7` |
| 分类损失 | 未加权 CrossEntropyLoss |
| Dropout | 0.5 |
| 分类阈值 | 0.5 |
| 主要检查点 | `best_loss` |

归一化和数据增强直接使用共享 Head40 loader 中的固定配置，保证与 Stem733 对照实验一致。

## 8. 五折训练

在项目根目录执行：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem333_multiboard_5fold.py --folds all
```

如需显式指定 PyTorch Python：

```powershell
& "D:\anconda\envs\pytorch-2.7.1-gpu\python.exe" `
  .\train_val\main_resnet18_head40_toml_direction_unified_stem333_multiboard_5fold.py `
  --folds all
```

默认模型输出目录：

```text
model_best_last/resnet18_head40_toml_direction_unified_stem333_multiboard_22boards_5fold
```

默认训练结果目录：

```text
train_val_result/resnet18_head40_toml_direction_unified_stem333_multiboard_22boards_5fold
```

每折独立保存：

```text
best_f1_resnet18_head40_toml_direction_unified_stem333.pth
best_loss_resnet18_head40_toml_direction_unified_stem333.pth
last_resnet18_head40_toml_direction_unified_stem333.pth
```

默认拒绝覆盖已有检查点。如确认重跑，必须显式增加：

```text
--allow-overwrite
```

## 9. 训练阶段比较规则

先完成全部五折，然后使用：

```text
fivefold_mean_std.csv
oof_metrics_best_loss.csv
```

主要比较 Stem333 与 Stem733 的：

- best_loss 五折验证损失
- best_loss 五折 Accuracy、Recall、F1、AUC 和 AP 均值与标准差
- best_loss OOF 指标
- 五折稳定性

在查看锁定三板之前，根据五折结果判断 `3×3×3` 是否值得保留。

## 10. 锁定三板测试

只有在五折训练全部完成，并且比较规则已经固定后，才执行：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem333_multiboard_3boards.py `
  --checkpoint-kinds best_loss
```

默认测试输出目录：

```text
test_result/resnet18_head40_toml_direction_unified_stem333_multiboard_22boards_5fold_locked_3boards
```

测试脚本会严格检查：

- 检查点实验名必须属于 Stem333 实验
- 模型类型必须为 `ResNet18Head40TOMLDirectionUnifiedStem333`
- `stem_conv_kernel` 必须为 `[3,3,3]`
- 方向统一配置必须一致
- 归一化配置必须一致
- 类别映射必须一致

因此旧 Stem733 检查点不能被误加载到本实验中。

## 11. 实验解释边界

如果 Stem333 优于 Stem733，只能说明在当前 Head40 数据、固定五折和当前训练方案下，较局部的起始卷积更合适。

如果二者接近，应同时考虑五折波动、训练时间和结构解释性，不能仅根据单折最高值判断。

本实验不改变裁剪长度、方向规则、残差层数、最终特征图尺寸或分类头，因此可以作为单变量 stem 卷积核消融实验。
