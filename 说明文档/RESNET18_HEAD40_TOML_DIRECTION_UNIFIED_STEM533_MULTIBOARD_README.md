# Head40 统一方向 ResNet18 Stem533 多板五折实验说明

## 1. 实验目的

本实验在已经固定的 Head40、方向统一和五折划分条件下，仅把 ResNet18 起始卷积核设置为：

```text
5 × 3 × 3
```

它用于补齐 `3×3×3`、`5×3×3`、`7×3×3` 三组 stem 卷积核消融，判断 Head40 输入在钻孔深度 D 方向需要多长的浅层感受范围。

除 stem 卷积核的 D 方向长度及对应 padding 外，本实验不改变池化、残差结构、特征图尺寸、五折清单、数据加载、方向统一、归一化、增强、训练参数、检查点选择规则和外部测试流程。

## 2. 本实验的四个新文件

Stem533 使用以下四个独立文件：

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_MULTIBOARD_README.md
```

模型、训练和测试入口均使用 Stem533 独立命名。训练脚本和测试脚本都导入本实验的新 Stem533 模型文件，并共享既有的 Head40 公共数据加载器；它们不会导入、执行或引用 Stem333、Stem733 的模型、训练或测试脚本。

本实验复用以下公共数据加载器和已经固定的数据资产：

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold
multiboard_head40_toml
test_3boards_head40_toml
```

依赖关系为：

```text
新 Stem533 训练脚本 ─┬→ 新 Stem533 模型文件
                     └→ 公共 Head40 数据加载器

新 Stem533 测试脚本 ─┬→ 新 Stem533 模型文件
                     └→ 公共 Head40 数据加载器
```

公共 loader 继续负责 Head40 RAW 读取、清单解析、方向统一、归一化、增强和按形状组批。固定五折清单不重建，也不需要重新运行任何 `make` 文件。

## 3. 与 Stem333、Stem733 的唯一结构差异

| 实验 | Stem kernel | Stem padding | Stem stride | Stem 后尺寸 | 模型参数量 |
|---|---|---|---|---|---:|
| Stem333 | `(3,3,3)` | `(1,1,1)` | `(1,1,1)` | 不变 | 33,140,802 |
| **Stem533** | **`(5,3,3)`** | **`(2,1,1)`** | **`(1,1,1)`** | **不变** | **33,141,954** |
| Stem733 | `(7,3,3)` | `(3,1,1)` | `(1,1,1)` | 不变 | 33,143,106 |

三组实验均保持：

```text
MaxPool3d(kernel=3×3×3, stride=1, padding=1)
Layer1/2/3/4 = [2,2,2,2]
通道数 = 64 → 128 → 256 → 512
Layer2、Layer3、Layer4 的首个残差块 stride=2
AdaptiveAvgPool3d(1×1×1)
Dropout(0.5)
Linear(512→2)
```

因此该实验比较的是首层在 D 方向观察 3、5、7 层体素时的效果，不是比较模型深度、输出特征图大小或分类头。

## 4. 固定数据与五折清单

训练和五折验证数据：

- 22 块板；
- 4,364 个样本；
- normal 3,181 个；
- defective 1,183 个；
- 五个验证折分别为 872、873、874、873、872 个样本。

锁定外部测试数据：

- 3 块板；
- 778 个样本；
- normal 556 个；
- defective 222 个。

必须直接使用：

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold/
├── split_config.json
├── all_samples_with_validation_fold.csv
├── locked_test_manifest.csv
├── fold_0/
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

本实验不重新分折。Stem333、Stem533、Stem733 的同一折必须具有完全相同的训练 SampleID 和验证 SampleID。

## 5. Head40、方向统一和数据处理

输入顺序为：

```text
N × C × D × H × W
```

其中输入通道为 1，深度固定为 40，横截面支持 `29×29`、`37×37`、`49×49`。

Head40 由 TOML `center2` 和 `bHeadUp` 确定。数据加载时将钻头统一到 D 轴高索引侧：

- `bHeadUp=false`：保持 D 方向；
- `bHeadUp=true`：沿 D 轴反转；
- 方向统一发生在增强和归一化之前；
- 不修改物理 RAW 文件；
- 数据增强不随机翻转 D 轴。

归一化和数据增强继续使用共享 Head40 loader 中的固定设置，保证三组 stem 实验的输入处理一致。

## 6. Stem533 完整模型结构

```text
Input: 1×40×H×W
  → Conv3d(1→64, kernel=5×3×3, stride=1, padding=2×1×1)
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

每个 `BasicBlock3D` 仍包含两个普通 `3×3×3` 卷积。尺寸或通道变化时，残差支路使用 `1×1×1` 卷积匹配。

模型类与关键 metadata 为：

```text
model = ResNet18Head40TOMLDirectionUnifiedStem533
stem_conv_kernel = [5, 3, 3]
stem_conv_stride = [1, 1, 1]
stem_pool_kernel = [3, 3, 3]
stem_pool_stride = [1, 1, 1]
direction_standardized = true
standardized_head_side = high_depth_index
```

## 7. 特征图形状

由于 Stem533 使用奇数卷积核、匹配 padding 和 stride 1，stem 不改变 D/H/W 尺寸。三种输入的最终特征图与 Stem333、Stem733 完全相同。

| 输入体积 `C×D×H×W` | Stem/Layer1 | Layer2 | Layer3 | Layer4 最终特征图 |
|---|---|---|---|---|
| `1×40×29×29` | `64×40×29×29` | `128×20×15×15` | `256×10×8×8` | `512×5×4×4` |
| `1×40×37×37` | `64×40×37×37` | `128×20×19×19` | `256×10×10×10` | `512×5×5×5` |
| `1×40×49×49` | `64×40×49×49` | `128×20×25×25` | `256×10×13×13` | `512×5×7×7` |

经过 `AdaptiveAvgPool3d(1)` 后均变为 `512×1×1×1`，再进入二分类层。

## 8. 默认训练设置

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

`best_f1` 和 `last` 可以作为补充结果保存，但论文主比较与外部测试应保持使用 `best_loss`，不能根据外部三板结果临时切换检查点。

## 9. 运行前自检

在项目根目录先执行：

```powershell
python -m py_compile `
  .\model\resnet18_3d_head40_toml_direction_unified_stem533_multiboard.py `
  .\train_val\main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold.py `
  .\test\test_resnet18_head40_toml_direction_unified_stem533_multiboard_3boards.py
```

再检查模型形状与参数量：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_multiboard.py
```

预期模型参数量为：

```text
33,141,954
```

预期三种输入分别输出：

```text
1×1×40×29×29 → 1×512×5×4×4 → 1×2
1×1×40×37×37 → 1×512×5×5×5 → 1×2
1×1×40×49×49 → 1×512×5×7×7 → 1×2
```

正式运行前还应确认：

- 固定五折目录包含 `split_config.json`、五个 fold 目录和 `locked_test_manifest.csv`；
- 实际训练数据根目录指向 `multiboard_head40_toml`；
- 实际测试数据根目录指向 `test_3boards_head40_toml`；
- 模型 metadata 中的 kernel 为 `[5,3,3]`；
- 旧实验输出目录不会被本实验使用；
- 若 Stem533 输出目录已经存在，先确认是否确实要重跑。

## 10. 五折训练命令

在项目根目录执行：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold.py `
  --folds all
```

如果自动发现不到数据目录，显式指定 Head40 数据根目录：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold.py `
  --folds all `
  --dataset-root "F:\325_275_and_sphere_data\钻孔数据325_275\multiboard_head40_toml"
```

默认模型输出目录：

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_multiboard_22boards_5fold
```

默认训练结果目录：

```text
train_val_result/resnet18_head40_toml_direction_unified_stem533_multiboard_22boards_5fold
```

每折独立保存：

```text
best_f1_resnet18_head40_toml_direction_unified_stem533.pth
best_loss_resnet18_head40_toml_direction_unified_stem533.pth
last_resnet18_head40_toml_direction_unified_stem533.pth
```

脚本默认拒绝覆盖已有检查点。只有明确确认重跑本实验时才增加：

```text
--allow-overwrite
```

## 11. 五折结果比较

完成全部五折后，优先查看训练结果目录中的：

```text
fivefold_mean_std.csv
oof_metrics_best_loss.csv
```

三组实验按完全相同规则比较：

- `best_loss` 的五折验证 loss 均值和标准差；
- `best_loss` 的 Accuracy、Precision、Recall、Specificity、F1、AUC、AP；
- `best_loss` 的 OOF 总体指标；
- 五折之间的波动，不能只看最好的一折；
- 参数量和训练耗时仅作补充，因为三组参数差异很小。

结果解释建议：

- `333 < 533 < 733`：增加浅层 D 方向范围持续有效；
- `333 < 533 ≈ 733`：5 层左右可能已经接近饱和；
- `533 > 733 > 333`：中等 D 方向范围可能更合适；
- 三者差异落在五折波动内：不能声称某个 kernel 明显更优。

比较完 `333/533/733` 后应冻结 stem，不继续扩展更多核尺寸。

## 12. 锁定三板外部测试

先完成五折训练并固定使用 `best_loss`，再执行本实验的新测试脚本：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_multiboard_3boards.py `
  --checkpoint-kinds best_loss
```

如果自动发现不到测试数据目录，显式指定：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_multiboard_3boards.py `
  --checkpoint-kinds best_loss `
  --test-dataset-root "F:\325_275_and_sphere_data\钻孔数据325_275\test_3boards_head40_toml"
```

默认读取模型目录：

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_multiboard_22boards_5fold
```

默认测试输出目录：

```text
test_result/resnet18_head40_toml_direction_unified_stem533_multiboard_22boards_5fold_locked_3boards
```

外部测试主要查看：

```text
fivefold_mean_std.csv
ensemble_metrics_by_board.csv
ensemble_predictions.csv
```

论文总体表现优先看 `board=ALL` 的 `best_loss` 指标，同时保留三块板各自指标和混淆矩阵，用于判断跨板稳定性。

测试脚本应严格拒绝不匹配检查点，至少检查：

- experiment 必须为 Stem533 实验名；
- 模型类必须为 `ResNet18Head40TOMLDirectionUnifiedStem533`；
- `stem_conv_kernel` 必须为 `[5,3,3]`；
- 五折检查点必须齐全；
- 方向统一、归一化和类别映射必须与训练一致；
- 锁定清单必须为 778 个样本、3 块板。

外部三板只用于最终报告，不用于重新选择 checkpoint、调整阈值或继续搜索 stem 尺寸。

## 13. 实验边界

Stem533 是单变量的 stem kernel 消融实验。无论结果如何，本实验都不能被解释为改变了 ResNet18 主体结构，也不能证明任意更长或更短卷积核在其他裁剪长度上同样有效。

本实验完成后，采用相同五折 `best_loss` 结果比较 Stem333、Stem533、Stem733，再报告锁定三板总体和分板表现即可。
