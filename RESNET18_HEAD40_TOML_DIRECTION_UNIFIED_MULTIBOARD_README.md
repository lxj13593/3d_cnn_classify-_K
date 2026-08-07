# TOML Head40 统一方向多板 3D ResNet18 实验说明

## 1. 实验范围

本说明只对应一套实验：使用 TOML `center2` 与 `bHeadUp` 生成的 Head40 预裁剪体积，在数据加载阶段统一钻孔方向，然后进行 22 板五折 3D ResNet18 训练，最后在锁定的 3 块外部板上测试。

实验名称：

```text
resnet18_head40_toml_direction_unified_multiboard_22boards_5fold
```

类别定义固定为：

```text
normal    = 0
defective = 1
```

本套代码不会修改原始 RAW，不会在训练或测试时重新裁剪，也不会把不同横截面插值成相同尺寸。所有清单、模型、训练结果和测试结果均写入本实验的独立目录。

## 2. 数据规模

### 2.1 训练与验证池

- 板数：22
- 样本总数：4,364
- 正常：3,181
- 缺陷：1,183
- Head40 RAW 尺寸分布：
  - `29×29×40`：164
  - `37×37×40`：3,943
  - `49×49×40`：257

五折是孔级固定五折。每一折的训练集和验证集都包含全部 22 块板，验证样本数依次为：

```text
fold_0: 872
fold_1: 873
fold_2: 874
fold_3: 873
fold_4: 872
```

### 2.2 锁定外部测试集

- 板号：0024、0030、0031
- 板数：3
- 样本总数：778
- 正常：556
- 缺陷：222
- RAW 尺寸：全部为 `37×37×40`

三块板的样本数分别为 130、116、532。锁定测试集与 22 板训练/验证池没有 SampleID 交集。

## 3. TOML Head40 裁剪与方向统一

### 3.1 Head40 输入的形成规则

本实验读取已经按 TOML 裁好的 Head40 RAW。TOML 序号与文件序号的对应关系为：

```text
toml_sequence = file_sequence - 1
```

先用非负数 half-up 规则把 TOML `center2` 转成整数索引：

```text
center_index = floor(center2 + 0.5)
```

再以 `center_index` 为中心执行：

- `bHeadUp=false`：`[center_index-15, center_index+25)`
- `bHeadUp=true`：`[center_index-25, center_index+15)`

两个区间长度均为 40。裁剪时根据 `bHeadUp` 向钻头一侧多保留 10 层，但预裁剪 RAW 本身仍保留原来的钻孔方向。

### 3.2 统一方向规则

统一目标是让所有样本的钻头都位于 D 轴高索引端：

- `bHeadUp=false`：原方向已经满足目标，保持不变。
- `bHeadUp=true`：在内存中沿 D 轴反转。

方向统计：

| 数据范围 | 保持原方向 | 沿 D 轴反转 |
| --- | ---: | ---: |
| 训练/验证池 | 4,293 | 71 |
| 锁定外部测试集 | 769 | 9 |

方向反转发生在数据加载阶段，不会回写物理 RAW。训练加载顺序为：

```text
读取预裁剪 Head40 RAW
→ 按 bHeadUp 决定是否沿 D 轴反转
→ 仅训练集执行 H/W 几何增强
→ 使用该样本 Head40 ROI 的 P1/P99 做归一化并裁剪到 [0, 1]
→ 仅训练集执行强度增强
→ 输入模型
```

P1/P99 统计来自每个样本自己的完整 Head40 ROI；D 轴反转不会改变其分位数。验证集和外部测试集不做数据增强。训练增强也不随机翻转 D 轴，因此不会破坏统一后的钻孔方向。

## 4. 固定五折与清单审计

统一方向清单目录为：

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold/
```

其中包含：

```text
all_samples_with_validation_fold.csv
locked_test_manifest.csv
fold_0/train.csv
fold_0/val.csv
...
fold_4/train.csv
fold_4/val.csv
fold_board_summary.csv
split_config.json
```

清单生成器直接读取 Head40 训练集和锁定测试集的 `source_audit.csv`，不依赖任何旧 `datasets/...5fold` 目录、固定划分 CSV 或旧划分脚本。

训练池按“板号 + 标签 + `FullVolumeRawShape`”分组。每组先按 SampleID 排序，再使用固定种子 42 的确定性算法分配 `validation_fold`；余数分配同时平衡分板标签数、分板总数、完整体积尺寸标签数、类别总数和折总数。完整体积统一方向清单使用同一组 SampleID、完整体积尺寸和算法，因此两套实验会得到逐 SampleID 完全一致的五折归属。

生成清单时会核对：

- SampleID 唯一性及训练/测试隔离
- 板号、序号、标签和标签索引
- Head40 RAW 尺寸及清单中的 SHA256 字段
- `BHeadUp`、原钻头侧和原数据未翻转状态
- 4,364 个训练/验证样本、22 块板及类别数
- 778 个锁定测试样本、3 块板及类别数
- 五折覆盖完整且每折训练/验证无 SampleID 交集

新清单额外记录：

```text
b_head_up
original_head_side
direction_flip_required
direction_standardized
standardized_head_side
```

## 5. 本实验专属文件

以下文件完整实现本套清单、加载、模型、训练和测试流程：

```text
data_operate/make_resnet18_head40_toml_direction_unified_multiboard_5fold.py
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py
model/resnet18_3d_head40_toml_direction_unified_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_multiboard_5fold.py
train_val/run_resnet18_head40_toml_direction_unified_multiboard_5fold.ps1
test/test_resnet18_head40_toml_direction_unified_multiboard_3boards.py
test/run_resnet18_head40_toml_direction_unified_multiboard_3boards.ps1
```

各文件职责：

- `make_..._5fold.py`：独立完成数据根目录解析、方向元数据核对、确定性五折分配、清单审计和输出。
- `data_load_...py`：读取 Head40 RAW，在内存中统一 D 轴方向，完成增强、归一化和同尺寸组批。
- `resnet18_3d_...py`：完整定义本实验使用的 Head40 3D ResNet18。
- `main_..._5fold.py`：完成五折训练、验证、检查点保存、OOF 汇总和结果绘图。
- 训练 PowerShell：先生成并核验本实验清单，再启动指定折训练。
- `test_..._3boards.py`：严格加载本实验五折检查点并评估锁定三板。
- 测试 PowerShell：先重新核验本实验清单，再启动锁定三板测试。

Python 文件之间只导入本实验专属的数据加载器和模型定义；PowerShell 入口也只执行本实验专属的 Python 文件。

## 6. 数据加载与增强

数据类型为 `uint8`，单通道输入形状为 `C×D×H×W`。不同横截面尺寸不会混在同一个 batch 中，而是由 `ShapeBatchSampler` 按 `D×H×W` 分组后组批。

归一化固定为逐样本 Head40 ROI 的 P1/P99：

```text
x = clip((x - P1) / (P99 - P1 + 1e-6), 0, 1)
```

仅训练集使用以下随机增强：

- H 或 W 随机翻转，概率 0.5
- H/W 平面随机旋转 90°、180°或 270°，概率 0.5
- H/W 平移 `[-3, 3]` 体素，概率 0.5
- 高斯噪声，概率 0.5，标准差范围 `[0.01, 0.03]`
- 亮度缩放，概率 0.5，范围 `[0.9, 1.1]`
- 对比度缩放，概率 0.5，范围 `[0.9, 1.1]`

除按 `bHeadUp` 执行的确定性方向统一外，不进行任何随机 D 轴翻转。

## 7. 模型结构

模型为单通道、二分类的 Head40 3D ResNet18：

- BasicBlock3D 数量：`[2, 2, 2, 2]`
- 四个 stage 通道数：`[64, 128, 256, 512]`
- stem 卷积：kernel `(7,3,3)`，stride `(1,1,1)`，padding `(3,1,1)`
- stem max-pool：kernel `(3,3,3)`，stride `(1,1,1)`，padding `(1,1,1)`
- 四个残差 stage 的首块步长：`[1,2,2,2]`
- 分类头：`AdaptiveAvgPool3d(1,1,1) → Dropout(0.5) → Linear(512,2)`

stem 不在深度方向提前下采样，以保留 Head40 的短深度信息。模型支持 `29×29×40`、`37×37×40` 和 `49×49×40` 三种输入尺寸。

## 8. 训练设置

默认参数：

| 项目 | 设置 |
| --- | --- |
| 折数 | 5 |
| 折种子 | `42, 123, 2026, 3407, 777` |
| Epoch | 50 |
| Batch size | 4 |
| DataLoader workers | 0 |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-3` |
| 学习率调度 | CosineAnnealingLR |
| 最小学习率 | `1e-7` |
| 损失函数 | 未加权 CrossEntropyLoss |
| Dropout | 0.5 |
| 分类阈值 | 0.5 |
| AMP | CUDA 可用时默认开启 |

每折分别保存三个检查点：

- `best_loss`：验证交叉熵最低的 epoch，主检查点。
- `best_f1`：验证 F1 最高的 epoch；F1 相同时选择验证损失更低者。
- `last`：第 50 个 epoch，或命令指定的最后一个 epoch。

主实验比较应使用 `best_loss`，不能把 `best_f1`、`best_loss` 和 `last` 混成同一结果。

## 9. 训练运行方法

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_head40_toml_direction_unified_multiboard_5fold.ps1
```

该入口会先生成并审计统一方向清单，然后依次训练五折。默认不覆盖已存在的检查点；只有确认要重跑时才可显式增加 `-AllowOverwrite`。

常用参数示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_head40_toml_direction_unified_multiboard_5fold.ps1 `
  -Folds all `
  -Epochs 50 `
  -BatchSize 4 `
  -NumWorkers 0
```

如果自动搜索不到数据集，可显式传入：

```powershell
-TrainDatasetRoot "完整的 multiboard_head40_toml 目录" `
-TestDatasetRoot "完整的 test_3boards_head40_toml 目录"
```

也可设置环境变量：

```text
DRILL_HEAD40_TOML_TRAIN_ROOT
DRILL_HEAD40_TOML_TEST_ROOT
```

模型目录：

```text
model_best_last/resnet18_head40_toml_direction_unified_multiboard_22boards_5fold/
```

训练与验证结果目录：

```text
train_val_result/resnet18_head40_toml_direction_unified_multiboard_22boards_5fold/
```

## 10. 训练输出

每折模型目录保存：

```text
best_f1_resnet18_head40_toml_direction_unified.pth
best_loss_resnet18_head40_toml_direction_unified.pth
last_resnet18_head40_toml_direction_unified.pth
```

每折结果包括训练历史、训练曲线、运行参数、数据统计，以及三个检查点各自的验证预测、指标、混淆矩阵、ROC 和 PR 图。

五折根目录的主要汇总文件为：

```text
fivefold_checkpoint_metrics.csv
fivefold_checkpoint_metrics.xlsx
fivefold_mean_std.csv
fivefold_mean_std.xlsx
oof_predictions_best_loss.csv
oof_metrics_best_loss.csv
oof_metrics_best_loss.json
oof_metrics_best_loss_by_board.csv
oof_metrics_best_loss_by_board_shape.csv
```

训练阶段首先查看 `fivefold_mean_std.csv` 中的 `best_loss` 列，以及完整 4,364 样本的 `oof_metrics_best_loss.csv`。指标包括 loss、accuracy、precision、recall、F1、specificity、AUC、AP 和混淆矩阵计数。

## 11. 锁定三板测试

必须在五折训练全部完成、模型选择规则和阈值已经固定后，才运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\test\run_resnet18_head40_toml_direction_unified_multiboard_3boards.ps1
```

测试入口要求所选检查点类型的五折文件全部存在，并核对检查点中的：

- 实验名称
- 模型类型及结构元数据
- 类别映射
- Head40 P1/P99 归一化参数
- 统一方向规则与目标钻头侧

测试阶段：

- 使用与训练相同的 `bHeadUp=true → flip D` 规则。
- 不做随机增强。
- 不做运行时裁剪、插值或重新训练。
- 默认阈值固定为 0.5。
- 分别报告 `best_f1`、`best_loss` 和 `last`，主结果仍为 `best_loss`。
- 同时给出每折结果及五折缺陷概率平均后的集成结果。

测试结果目录：

```text
test_result/resnet18_head40_toml_direction_unified_multiboard_22boards_5fold_locked_3boards/
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

## 12. 锁定测试约束

0024、0030、0031 三块外部板只用于最终泛化评估，严禁用于：

- 五折划分
- 训练或验证
- 归一化总体统计；本实验只用各样本自身 Head40 ROI 的 P1/P99
- epoch 或检查点选择
- 模型结构、裁剪范围或增强策略选择
- 阈值调节
- 根据测试结果反复修改方案后继续把同一测试集称为“锁定测试集”

正式顺序固定为：

```text
生成并审计统一方向清单
→ 完成全部五折训练
→ 以 best_loss 五折与 OOF 结果完成训练阶段比较
→ 固定方案和阈值
→ 运行一次锁定三板测试
→ 以 best_loss 为主报告外部泛化结果
```

## 13. 结果解释边界

这套结果代表“按 TOML `center2 + bHeadUp` 预裁剪 Head40、运行时统一方向、使用 Head40 专用 stem”的整体方案。它不能被解释成只改变了一个裁剪参数，也不能把孔级五折验证结果等同于完全未见板的泛化结果；新板泛化应以最后的锁定三板测试为准。
