# Head40-733 Layer1 首层非对称卷积多板五折实验

## 1. 实验目的

本实验以当前 Head40-733、方向统一的 3D ResNet18 为基线，只修改一个位置：

```text
layer1[0].conv1
```

原来的单个 `3×3×3 Conv + BN` 替换为自定义 **3D Planar ACB**：

```text
输入
 ├─ 3×3×3 Conv + BN ─┐
 ├─ 1×3×3 Conv + BN ─┤
 ├─ 3×1×3 Conv + BN ─┼─ 相加 → BasicBlock 原有 ReLU
 └─ 3×3×1 Conv + BN ─┘
```

四个分支分别做 BatchNorm 后再相加。完整 `3×3×3` 分支保留，用于保留三维立方邻域信息；三个平面分支用于增强不同方向的局部特征。当前实现用于训练和测试，**不包含部署阶段的卷积重参数化/分支融合**。

## 2. 本实验的四个新文件

```text
model/resnet18_3d_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_LAYER1_BLOCK1_CONV1_ACB_MULTIBOARD_README.md
```

模型、训练、测试和说明均使用独立命名，不修改旧实验文件。训练和测试继续共享现有 Head40 数据加载器：

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py
```

## 3. 唯一结构变化

基线与本实验的差别只有 `layer1[0].conv1`：

| 位置 | Head40-733 基线 | 本实验 |
|---|---|---|
| Stem | `7×3×3 Conv + BN + ReLU` | 不变 |
| MaxPool | `3×3×3, stride=1` | 不变 |
| `layer1[0].conv1` | `3×3×3 Conv + BN` | 四分支 3D Planar ACB |
| `layer1[0].conv2` | `3×3×3 Conv + BN` | 不变 |
| `layer1[1]` | 两个普通 `3×3×3` 卷积 | 不变 |
| Layer2/3/4、池化、Dropout、FC | 基线设置 | 不变 |

完整结构为：

```text
Input: 1×40×H×W
  → Stem: Conv3d(1→64, kernel=7×3×3, stride=1, padding=3×1×1)
  → MaxPool3d(kernel=3×3×3, stride=1, padding=1)
  → Layer1 Block1: ACB conv1 → 普通 3×3×3 conv2
  → Layer1 Block2: 两个普通 3×3×3 卷积
  → Layer2: 2 个普通 BasicBlock3D，首块 stride=2
  → Layer3: 2 个普通 BasicBlock3D，首块 stride=2
  → Layer4: 2 个普通 BasicBlock3D，首块 stride=2
  → AdaptiveAvgPool3d(1)
  → Dropout(0.5)
  → Linear(512→2)
```

关键模型元数据：

```text
model = ResNet18Head40TOMLDirectionUnifiedLayer1Block1Conv1ACB
acb_location = layer1.0.conv1
acb_branch_kernels = [[3,3,3], [1,3,3], [3,1,3], [3,3,1]]
acb_branch_batchnorm_before_sum = true
acb_reparameterization_implemented = false
```

## 4. 保持不变的实验条件

- 输入仍为 TOML `center2` 和 `bHeadUp` 得到的 Head40，尺寸为 `1×40×H×W`。
- 钻头方向仍统一到 D 轴高索引侧。
- 继续使用原来的归一化、数据增强和按形状组批逻辑。
- 继续使用同一套固定 22 板五折清单，不重新划分，不需要运行新的 make 文件。
- 五折随机种子仍为 `42, 123, 2026, 3407, 777`。
- Epoch、batch size、AdamW、学习率、权重衰减、CosineAnnealingLR、损失函数、阈值均与 Head40-733 基线一致。
- 锁定三块外部板仍为 778 个样本，只在五折训练完成并锁定 checkpoint 后测试。

固定清单位置：

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold
```

## 5. 参数量与特征图

预期可训练参数量：

```text
33,254,082
```

ACB 不改变 padding、stride、通道数或空间尺寸，因此最终特征图与 Head40-733 基线相同：

| 输入体积 `C×D×H×W` | Layer4 最终特征图 |
|---|---|
| `1×40×29×29` | `512×5×4×4` |
| `1×40×37×37` | `512×5×5×5` |
| `1×40×49×49` | `512×5×7×7` |

AdaptiveAvgPool3d 后均为 `512×1×1×1`，分类输出为 2 类。

## 6. 运行前检查

在项目根目录执行：

```powershell
python -m py_compile `
  .\model\resnet18_3d_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard.py `
  .\train_val\main_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_5fold.py `
  .\test\test_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_3boards.py

python .\model\resnet18_3d_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard.py
```

## 7. 五折训练

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_5fold.py `
  --folds all
```

如果未自动找到数据，显式指定训练数据根目录：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_5fold.py `
  --folds all `
  --dataset-root "F:\325_275_and_sphere_data\钻孔数据325_275\multiboard_head40_toml"
```

实验名：

```text
resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_22boards_5fold
```

默认 checkpoint 目录：

```text
model_best_last/resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_22boards_5fold
```

每折保存：

```text
best_f1_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb.pth
best_loss_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb.pth
last_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb.pth
```

默认训练结果目录：

```text
train_val_result/resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_22boards_5fold
```

论文主比较固定使用 **best_loss**。验证集主要查看 `fivefold_mean_std.csv` 和 `oof_metrics_best_loss.csv`；`best_f1` 与 `last` 仅作补充，不能根据外部测试结果临时更换 checkpoint。

## 8. 锁定三板外部测试

五折训练完成后执行：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_3boards.py `
  --checkpoint-kinds best_loss
```

如果未自动找到外部数据，显式指定：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_3boards.py `
  --checkpoint-kinds best_loss `
  --test-dataset-root "F:\325_275_and_sphere_data\钻孔数据325_275\test_3boards_head40_toml"
```

默认测试结果目录：

```text
test_result/resnet18_head40_toml_direction_unified_layer1_block1_conv1_acb_multiboard_22boards_5fold_locked_3boards
```

主要结果文件：

```text
fivefold_mean_std.csv
fivefold_mean_std.xlsx
ensemble_metrics_by_board.csv
ensemble_predictions.csv
```

论文中保持与验证集一致，主报 `best_loss` 的五折均值和标准差；外部三板总体表现查看 `fivefold_mean_std.csv` 中 `Board=ALL` 的结果。五折概率集成结果可以作为补充，不能用它反向选择模型。

## 9. 公平比较要求

本实验只回答一个问题：在 Head40-733 基线上，仅增强 `layer1[0].conv1` 的方向性局部表征是否有效。比较时必须同时满足：

- 使用完全相同的五折 SampleID；
- 使用相同训练参数和 checkpoint 选择规则；
- 使用相同的锁定三板测试清单；
- 主比较均为 `best_loss`；
- 不把参数增加、其他层变化或外部测试调参混入本实验。
