# Head40 Parallel-Stem333-511 多板五折实验说明

## 1. 实验目的

本实验以原始 Head40-533 为基线，只替换 Stem：

```text
原结构：5x3x3 Conv + BN -> ReLU

新结构：               ┌─ 3x3x3 Conv + BN ─┐
输入 ------------------┤                    ├─ 相加 -> ReLU
                       └─ 5x1x1 Conv + BN ─┘
```

`3x3x3` 分支提取局部三维特征，`5x1x1` 分支提取钻孔 D 方向连续特征。两个分支均输出 64 通道。

除 Stem 外，数据、方向统一、五折清单、训练参数、Layer2/3/4、池化和分类头均与原始 Head40-533 一致。本实验不使用 Layer2 stride122，Layer2 恢复标准 `stride=(2,2,2)`。

## 2. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_parallel_stem333_511_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_PARALLEL_STEM333_511_MULTIBOARD_README.md
```

训练和测试脚本只导入本实验的新模型文件，不引用其他实验的模型、训练或测试脚本。公共 Head40 数据加载器和既有五折清单继续使用。

## 3. 模型结构

```text
输入：1 x 40 x H x W
Parallel Stem：
  分支1：3x3x3 Conv，padding=(1,1,1)，stride=1，1->64 + BN
  分支2：5x1x1 Conv，padding=(2,0,0)，stride=1，1->64 + BN
  两分支相加 -> ReLU
MaxPool：3x3x3，stride=1
Layer1：2 x BasicBlock，64通道，stride=1
Layer2：2 x BasicBlock，128通道，首块 stride=(2,2,2)
Layer3：2 x BasicBlock，256通道，首块 stride=(2,2,2)
Layer4：2 x BasicBlock，512通道，首块 stride=(2,2,2)
AdaptiveAvgPool3d(1)
Dropout(0.5)
Linear(512,2)
```

最终特征图与 Head40-333、533、733 相同：

| 输入体积 | Layer4 特征图 |
|---|---|
| `1x40x29x29` | `512x5x4x4` |
| `1x40x37x37` | `512x5x5x5` |
| `1x40x49x49` | `512x5x7x7` |

## 4. 运行顺序

检查模型：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_parallel_stem333_511_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_5fold.py
```

训练完成后测试锁定三板，论文主结果使用 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 5. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_parallel_stem333_511_multiboard_22boards_5fold_locked_3boards
```

比较时使用相同五折的 `best_loss` 验证均值和锁定三板总体五折均值，不根据外部测试结果反向选择 checkpoint。
