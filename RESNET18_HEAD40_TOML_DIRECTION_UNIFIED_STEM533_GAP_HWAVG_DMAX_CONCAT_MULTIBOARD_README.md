# Head40-533 GAP + HWAvg-DMax 拼接池化实验说明

## 1. 实验内容

本实验以标准 Head40-533 ResNet18 为基线，只修改 Layer4 后的池化和全连接层：

```text
Layer4 feature [B,512,D,H,W]
        │
        ├─ AdaptiveAvgPool3d(1) ─────────> GAP [B,512] ─────┐
        │                                                   │
        └─ H/W Average [B,512,D] ─> D Max [B,512] ─────────┤
                                                            │
                                                       拼接 [B,1024]
                                                            │
                                                        Dropout(0.5)
                                                            │
                                                        FC 1024 -> 2
```

标准 GAP 提供全局平均信息，HWAvg-DMax 分支保留 D 方向最强响应。两个 512 维向量拼接，不进行相加。

除池化和 FC 输入维度外，Stem533、ResNet18 主体、数据、方向统一、五折清单、训练参数和锁定三板测试协议均保持不变。

## 2. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_GAP_HWAVG_DMAX_CONCAT_MULTIBOARD_README.md
```

模型、训练、测试、checkpoint 和输出目录均为本实验独立文件，不引用其他实验的模型、训练或测试脚本。公共 Head40 数据加载器和既有五折清单继续使用。

## 3. 模型结构

```text
输入：1 x 40 x H x W
Stem：5x3x3 Conv，stride=1
MaxPool：3x3x3，stride=1
Layer1：2 x BasicBlock，64通道，stride=1
Layer2：2 x BasicBlock，128通道，首块 stride=(2,2,2)
Layer3：2 x BasicBlock，256通道，首块 stride=(2,2,2)
Layer4：2 x BasicBlock，512通道，首块 stride=(2,2,2)
GAP分支：[B,512]
HWAvg-DMax分支：[B,512]
Concat：[B,1024]
Dropout(0.5)
Linear(1024,2)
```

Layer4 特征图仍为：

| 输入体积 | Layer4 特征图 |
|---|---|
| `1x40x29x29` | `512x5x4x4` |
| `1x40x37x37` | `512x5x5x5` |
| `1x40x49x49` | `512x5x7x7` |

新模型仅比标准 533 增加 1,024 个全连接层参数，总参数量为 33,142,978。

## 4. 运行顺序

检查模型：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_5fold.py
```

训练完成后测试锁定三板，论文主比较使用 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 5. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_gap_hwavg_dmax_concat_multiboard_22boards_5fold_locked_3boards
```

比较时使用标准 Head40-533、本实验以及 Head40-733 的 `best_loss` 五折验证结果和锁定三板总体五折均值。
