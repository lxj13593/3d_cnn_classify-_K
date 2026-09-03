# Head40-533 Layer2-Stride122 多板五折实验说明

## 1. 实验目的

本实验以 Head40-533 ResNet18 为唯一基线，只修改一处：

```text
Layer2 第一个 BasicBlock 的 stride
(2,2,2) -> (1,2,2)
```

深度方向 D 在 Layer2 保持不变，横截面 H/W 仍然下采样。其余模型结构、数据、五折划分、训练参数和外部三板测试规则全部保持不变。

## 2. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_layer2_stride122_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER2_STRIDE122_MULTIBOARD_README.md
```

训练和测试脚本只导入本实验的新模型文件，不引用其他训练或测试脚本。数据加载继续使用既有的 Head40 统一方向公共数据加载器和同一套五折清单。

## 3. 模型结构

```text
输入：1 x 40 x H x W
Stem：5x3x3 Conv，stride=1
MaxPool：3x3x3，stride=1
Layer1：2 x BasicBlock，64通道，stride=1
Layer2：2 x BasicBlock，128通道，首块 stride=(1,2,2)
Layer3：2 x BasicBlock，256通道，首块 stride=(2,2,2)
Layer4：2 x BasicBlock，512通道，首块 stride=(2,2,2)
AdaptiveAvgPool3d(1)
Dropout(0.5)
Linear(512, 2)
```

参数量与 Head40-533 基线相同，为 33,141,954；变化只来自下采样步幅。

## 4. 特征图尺寸

| 输入尺寸 | Layer2 | Layer3 | Layer4 |
|---|---|---|---|
| `40x29x29` | `40x15x15` | `20x8x8` | `10x4x4` |
| `40x37x37` | `40x19x19` | `20x10x10` | `10x5x5` |
| `40x49x49` | `40x25x25` | `20x13x13` | `10x7x7` |

最终 Layer4 特征图分别为：

```text
512 x 10 x 4 x 4
512 x 10 x 5 x 5
512 x 10 x 7 x 7
```

## 5. 运行顺序

先检查模型：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_layer2_stride122_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_5fold.py
```

训练完成后测试锁定三块外部板，论文主结果使用 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 6. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_layer2_stride122_multiboard_22boards_5fold_locked_3boards
```

与 Head40-533 基线比较时，使用相同五折的验证集 `best_loss` 平均结果，再比较五折模型在锁定外部三板上的总体平均表现。
