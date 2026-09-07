# Head40-533 HWAvg-DMax 多板五折实验说明

## 1. 实验内容

本实验以标准 Head40-533 ResNet18 为基线，只修改分类前的池化方式：

```text
标准533：Layer4 -> AdaptiveAvgPool3d(1) -> [B,512]

本实验：Layer4 [B,512,D,H,W]
              -> H/W维度平均 -> [B,512,D]
              -> D维度最大值 -> [B,512]
```

对应运算：

```python
x = x.mean(dim=(3, 4))
x = torch.amax(x, dim=2)
```

Stem 仍为 `5x3x3`，Layer2/3/4 首块仍为 `stride=(2,2,2)`。数据、方向统一、五折划分、训练参数和锁定三板测试规则均不改变。

## 2. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_HWAVG_DMAX_MULTIBOARD_README.md
```

模型类、实验名、checkpoint 名、训练输出和测试输出均采用独立命名。训练和测试脚本只导入本实验的新模型文件，不引用其他训练或测试脚本；公共 Head40 数据加载器和原有五折清单继续使用。

## 3. 完整结构

```text
输入：1 x 40 x H x W
Stem：5x3x3 Conv，stride=1
MaxPool：3x3x3，stride=1
Layer1：2 x BasicBlock，64通道，stride=1
Layer2：2 x BasicBlock，128通道，首块 stride=(2,2,2)
Layer3：2 x BasicBlock，256通道，首块 stride=(2,2,2)
Layer4：2 x BasicBlock，512通道，首块 stride=(2,2,2)
H/W Average
D Max
Dropout(0.5)
Linear(512,2)
```

Layer4 特征图仍为：

| 输入体积 | Layer4 特征图 |
|---|---|
| `1x40x29x29` | `512x5x4x4` |
| `1x40x37x37` | `512x5x5x5` |
| `1x40x49x49` | `512x5x7x7` |

池化后的分类向量统一为 `[B,512]`，因此全连接层和模型参数量与标准 Head40-533 相同。

## 4. 运行顺序

检查模型：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_5fold.py
```

训练完成后测试锁定三板，论文主比较使用 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 5. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_hwavg_dmax_multiboard_22boards_5fold_locked_3boards
```

比较时使用标准 Head40-533 与本实验的 `best_loss` 五折验证结果和锁定三板总体五折均值，不根据外部测试结果更换 checkpoint。
