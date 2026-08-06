# 完整体积统一钻孔方向 3D ResNet18：22 板五折训练与锁定三板测试

## 1. 实验目的

本实验使用钻孔的**完整三维体积**进行正常/缺陷二分类，并在数据加载阶段把所有钻头统一到输入深度轴 `D` 的高索引端。

本套实验拥有独立的清单生成、数据加载、模型、训练、测试和 PowerShell 启动文件。运行时不会调用其他实验的 Python 或 PowerShell 脚本，模型和结果也写入本实验专属目录。

类别定义固定为：

- `normal = 0`
- `defective = 1`

验证阶段以 `best_loss` 为主要检查点。锁定外部三板只在五折训练全部完成后测试，不参与训练、选模、阈值调整或结构选择。

## 2. 数据范围

### 2.1 训练/验证池

完整体积数据集名称：

```text
multiboard_binary_dataset_no_initial_board
```

数据规模：

- 板数：22
- 样本总数：4,364
- 正常：3,181
- 缺陷：1,183

完整体积 RAW 的 `W×H×D` 尺寸包括：

- `29×29×237`
- `37×37×271`
- `37×37×291`
- `37×37×296`
- `49×49×361`

训练与验证都来自这 4,364 个样本；没有内部测试集。

### 2.2 锁定外部测试集

完整体积测试数据集名称：

```text
new_data_test_3boards_0024_0030_0031
```

数据规模：

- 板号：0024、0030、0031
- 板数：3
- 样本总数：778
- 正常：556
- 缺陷：222
- RAW 尺寸：全部为 `37×37×296`

训练/验证池和锁定测试集之间的板号、SampleID 与 RAW SHA256 均应保持无交集。

## 3. 固定五折及方向清单的生成

本实验的新清单目录为：

```text
datasets/resnet18_multiboard_direction_unified_5fold/
```

五折归属不是重新随机划分。清单生成器读取下面已经锁定的 `SampleID → validation_fold` 数据：

```text
datasets/resnet18_multiboard_final_5fold/all_samples_with_validation_fold.csv
datasets/resnet18_multiboard_final_5fold/locked_test_manifest.csv
datasets/resnet18_multiboard_final_5fold/split_config.json
```

这里只继承固定的样本身份和折号，不执行该目录对应的任何脚本。这样可以保证完整体积统一方向实验与其他对照实验使用相同五折。

各折验证样本数固定为：

| 折 | 训练数 | 验证数 |
|---:|---:|---:|
| 0 | 3,492 | 872 |
| 1 | 3,491 | 873 |
| 2 | 3,490 | 874 |
| 3 | 3,491 | 873 |
| 4 | 3,492 | 872 |

每一折的训练集和验证集都覆盖全部 22 块板。

方向信息来自以下两个数据集各自的 `source_audit.csv`：

```text
multiboard_head40_toml
test_3boards_head40_toml
```

这里只读取 `bHeadUp` 以及身份审计字段，不读取 Head40 RAW 作为本实验模型输入。清单生成器会逐样本核对：

- SampleID
- 板号与孔序号
- 标签与标签索引
- 完整体积 RAW 尺寸
- 完整体积 RAW SHA256
- `bHeadUp` 与原始钻头所在侧

任一身份、尺寸或哈希不一致都会停止生成。生成后还会检查五折覆盖、训练/验证互斥、训练与锁定测试 SampleID 无交集，以及预期的数据规模。

生成的主要文件包括：

```text
all_samples_with_validation_fold.csv
locked_test_manifest.csv
fold_0/train.csv、fold_0/val.csv
...
fold_4/train.csv、fold_4/val.csv
fold_board_summary.csv
split_config.json
```

## 4. 钻孔方向统一规则

统一目标：**钻头全部位于输入 `D` 轴的高索引端**。

规则固定为：

- `bHeadUp=false`：原钻头位于高 `D` 端，保持原顺序。
- `bHeadUp=true`：原钻头位于低 `D` 端，仅沿 `D` 轴反转。

方向数量：

| 数据范围 | 保持原方向 | 沿 D 轴反转 |
|---|---:|---:|
| 训练/验证池 | 4,293 | 71 |
| 锁定外部测试 | 769 | 9 |

RAW 文件本身不会被改写，也不会生成一份翻转后的物理 RAW。反转仅在每次加载到内存后执行，位置在数据增强和归一化之前。

对于形状为 `C×D×H×W` 的张量，只翻转维度 `D`；训练增强不允许再次翻转 `D`，因此不会破坏统一后的钻头方向。

## 5. 完整体积数据处理流程

每个样本的处理顺序为：

```text
uint8 RAW
→ 按清单尺寸还原为 D×H×W
→ bHeadUp=true 时沿 D 轴反转
→ 训练集执行 H/W 几何增强，验证和测试不增强
→ 使用该样本完整体积自身的 P1/P99 归一化并截断到 [0,1]
→ 训练集执行强度增强
→ 送入 3D ResNet18
```

归一化固定为逐样本完整体积 P1/P99，不使用训练集全局统计量，也不从锁定测试集估计任何共享参数。

训练增强仅在训练集启用：

- H 或 W 单轴翻转：概率 0.5
- H/W 平面旋转 90°、180°或 270°：概率 0.5
- H/W 平移：概率 0.5，范围 `[-3,3]` 体素
- 高斯噪声：概率 0.5，标准差范围 `[0.01,0.03]`
- 亮度缩放：概率 0.5，范围 `[0.9,1.1]`
- 对比度缩放：概率 0.5，范围 `[0.9,1.1]`
- D 轴随机翻转：禁用

不同体积尺寸不会混入同一个 batch。数据加载器按 `D×H×W` 分组组 batch，因此无需插值、补齐或裁剪完整体积。

## 6. 模型结构

模型为本实验专属的完整体积 3D ResNet18：

- 输入通道：1
- BasicBlock3D 数量：`[2,2,2,2]`
- 主干通道：`[64,128,256,512]`
- stem 卷积核：`(7,3,3)`
- stem 卷积步长：`(2,1,1)`
- stem max-pool 步长：`(2,1,1)`
- 分类池化：`AdaptiveAvgPool3d(1,1,1)`
- Dropout：0.5
- 输出类别数：2

该 stem 会在长度方向下采样，但保持早期 H/W 分辨率；自适应全局池化允许上述多种完整体积尺寸使用同一个模型。

## 7. 训练设置

默认训练参数：

| 项目 | 设置 |
|---|---|
| 五折种子 | `42, 123, 2026, 3407, 777` |
| Epoch | 50 |
| Batch size | 4 |
| DataLoader workers | 0 |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-3` |
| 学习率调度 | CosineAnnealingLR |
| `T_max` | 当前训练 Epoch 数 |
| 最低学习率 | `1e-7` |
| 损失函数 | 未加权 CrossEntropyLoss |
| 分类阈值 | 0.5 |
| AMP | CUDA 可用时默认启用 |

每折保存三种检查点，并保持结果分开：

- `best_loss`：验证交叉熵最低的 epoch，主检查点。
- `best_f1`：验证 F1 最高的 epoch；F1 相同时取验证损失更低者。
- `last`：最后一个 epoch。

报告指标包括 loss、Accuracy、Precision、Recall、F1、Specificity、AUC-ROC、AP 及混淆矩阵计数。主要比较五折 `best_loss` 的均值/标准差以及 4,364 个样本组成的 `best_loss` OOF 指标。

## 8. 本套实验的专属文件

```text
data_operate/make_resnet18_multiboard_direction_unified_5fold.py
data_operate/data_load_resnet18_multiboard_direction_unified.py
model/resnet18_3d_multiboard_direction_unified.py
train_val/main_resnet18_multiboard_direction_unified_5fold.py
train_val/run_resnet18_multiboard_direction_unified_5fold.ps1
test/test_resnet18_multiboard_direction_unified_3boards.py
test/run_resnet18_multiboard_direction_unified_3boards.ps1
RESNET18_DIRECTION_UNIFIED_MULTIBOARD_README.md
```

文件之间只在本套实验内部配合：训练脚本导入本套数据加载器和本套模型，测试脚本也只加载本套模型检查点。

## 9. 正式运行顺序

所有命令都从项目根目录执行。

### 9.1 完成五折训练

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_multiboard_direction_unified_5fold.ps1
```

该启动器会先运行本套方向清单生成器，重新生成并审计统一方向清单；通过后再依次训练指定折。默认训练全部五折。

常用可选参数示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_multiboard_direction_unified_5fold.ps1 `
  -Folds all `
  -Epochs 50 `
  -BatchSize 4 `
  -NumWorkers 0 `
  -TrainDatasetRoot "F:\...\multiboard_binary_dataset_no_initial_board" `
  -OrientationTrainDatasetRoot "F:\...\multiboard_head40_toml" `
  -OrientationTestDatasetRoot "F:\...\test_3boards_head40_toml" `
  -PythonExe "C:\...\python.exe"
```

已有检查点时训练默认拒绝覆盖。只有确认要重跑相应折时才传入 `-AllowOverwrite`。

### 9.2 检查训练结果并锁定方案

确认以下条件后才能进入外部测试：

- 五个 fold 均训练完成。
- 每折的 `best_loss`、`best_f1`、`last` 检查点齐全。
- 主要验证结论基于 `best_loss`，阈值仍为 0.5。
- 不根据外部三板结果反向修改模型、训练参数、阈值或检查点选择规则。

### 9.3 运行锁定三板测试

```powershell
powershell -ExecutionPolicy Bypass -File .\test\run_resnet18_multiboard_direction_unified_3boards.ps1
```

测试启动器同样先用本套生成器重新生成并审计清单，然后要求五折检查点全部存在，最后在 778 个锁定样本上评估。

需要显式指定路径时可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\test\run_resnet18_multiboard_direction_unified_3boards.ps1 `
  -BatchSize 4 `
  -NumWorkers 0 `
  -TestDatasetRoot "F:\...\new_data_test_3boards_0024_0030_0031" `
  -OrientationTrainDatasetRoot "F:\...\multiboard_head40_toml" `
  -OrientationTestDatasetRoot "F:\...\test_3boards_head40_toml" `
  -PythonExe "C:\...\python.exe"
```

## 10. 输出目录

### 10.1 五折清单

```text
datasets/resnet18_multiboard_direction_unified_5fold/
```

### 10.2 模型检查点

```text
model_best_last/resnet18_multiboard_direction_unified_22boards_5fold/fold_0/
...
model_best_last/resnet18_multiboard_direction_unified_22boards_5fold/fold_4/
```

每折包含：

```text
best_loss_resnet18_3d_direction_unified.pth
best_f1_resnet18_3d_direction_unified.pth
last_resnet18_3d_direction_unified.pth
```

### 10.3 训练/验证结果

```text
train_val_result/resnet18_multiboard_direction_unified_22boards_5fold/
```

每折保存运行配置、数据摘要、训练历史、训练曲线，以及三种检查点各自的验证预测、指标、混淆矩阵、ROC 和 PR 图。

五折根目录主要输出：

```text
fivefold_checkpoint_metrics.csv/.xlsx
fivefold_mean_std.csv/.xlsx
oof_predictions_best_loss.csv
oof_metrics_best_loss.csv/.json
oof_metrics_best_loss_by_board.csv
oof_metrics_best_loss_by_board_shape.csv
experiment_config.json
```

### 10.4 锁定三板测试结果

```text
test_result/resnet18_multiboard_direction_unified_22boards_5fold_locked_3boards/
```

主要输出：

```text
predictions_all_folds.csv
fold_metrics_by_board.csv
ensemble_predictions.csv
ensemble_metrics_by_board.csv
fivefold_mean_std.csv
fivefold_mean_std_raw.csv
fivefold_mean_std.xlsx
test_config.json
plots/
```

测试结果分别报告 `best_f1`、`best_loss` 和 `last`，并分别计算五折单模型结果与五折概率平均集成结果；三种检查点不能混写，主解释仍以 `best_loss` 为准。

## 11. 锁定外部测试约束

外部三板测试阶段固定执行：

- 使用训练时相同的逐样本完整体积 P1/P99 归一化。
- 使用训练时相同的方向统一规则。
- 不进行数据增强。
- 不裁剪、不插值、不重新训练。
- 分类阈值保持 0.5。
- 不利用测试标签选择 epoch、模型结构或超参数。
- 不因测试结果不理想而重复挑选检查点。

只有遵守以上顺序和约束，锁定三板结果才能作为未见新板泛化能力的有效评估。
