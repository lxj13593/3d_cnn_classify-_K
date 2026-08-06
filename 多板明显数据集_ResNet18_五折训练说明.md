# 多板明显数据集 ResNet18 五折训练说明

## 1. 本次实验范围

- 数据源默认位置：`datasets\clear_binary_dataset`
- 样本总数：4403，其中正常 3262、明显缺陷 1141，共 15 块板。
- 模型：原始 3D ResNet18 基线，不含 ECA、Head ROI 或其他结构改动。
- 评估：五折交叉验证，每折约 80% 训练、20% 验证。
- 本轮不设置内部测试集，也不编写或运行外部测试。
- 数据盘仅被读取。RAW、二维图、原始清单均不复制、不改名、不删除。

## 2. 配套文件

| 文件 | 用途 |
|---|---|
| `data_operate/make_multiboard_clear_5fold.py` | 从当前数据盘的两类总清单生成五折 CSV，并执行重复、尺寸和泄漏审计 |
| `data_operate/data_load_multiboard_clear.py` | 读取多尺寸 RAW、逐样本 P1/P99 归一化、训练增强和同尺寸组 batch |
| `model/resnet18_3d_multiboard.py` | 本实验独立的原始 3D ResNet18 |
| `train_val/main_resnet18_multiboard_clear_5fold.py` | 五折训练、验证、检查点保存、OOF 与均值标准差汇总 |
| `train_val/run_resnet18_multiboard_clear_5fold.ps1` | 先重建并审计清单，再启动训练的一键脚本 |

五折清单已生成在：

`datasets/multiboard_clear_5fold`

这里只保存 CSV/JSON，不保存 RAW 副本。清单只保存 `raw_relative_path`，不保存带盘符的绝对数据路径，因此项目整体移动到另一台电脑后仍可按项目相对位置读取。

未指定数据目录时，脚本默认读取项目内的 `datasets\clear_binary_dataset`；也兼容把数据集两类目录直接放进 `datasets`，或放进 `datasets` 下唯一一个子文件夹。`--dataset-root`、PowerShell 的 `-DatasetRoot` 和环境变量 `DRILL_DATASET_ROOT` 可用于明确覆盖。项目内尚未放入数据时，脚本才会回退搜索旧外接盘位置。

## 3. 五折划分规则

1. 按“板号 + 类别 + 原始尺寸”分层，每层固定种子 42 打乱并尽量均分到五折。
2. 每个样本只指定一个 `validation_fold`。某折验证时，其余四折作为训练集。
3. 同一钻孔、同一路径和同一 RAW SHA256 不会同时进入训练与验证。
4. 所有样本恰好验证一次；五折合并后得到 4403 条 OOF 预测。
5. 每折训练集和验证集均包含全部 15 块板，但不强行让不同板的样本总数相等。
6. 只有 1–2 个缺陷的小组不可能覆盖五个验证折，按全局平衡轮流放置，绝不复制样本。

实际划分：

| Fold | 训练总数 | 训练正常 | 训练缺陷 | 验证总数 | 验证正常 | 验证缺陷 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3522 | 2610 | 912 | 881 | 652 | 229 |
| 1 | 3523 | 2610 | 913 | 880 | 652 | 228 |
| 2 | 3522 | 2609 | 913 | 881 | 653 | 228 |
| 3 | 3522 | 2609 | 913 | 881 | 653 | 228 |
| 4 | 3523 | 2610 | 913 | 880 | 652 | 228 |

## 4. 多尺寸输入

RAW 文件名和数据清单中的尺寸顺序是“宽 x 高 x 长”，PyTorch 输入顺序转换为“长 x 高 x 宽”。本数据集包含：

- 29 x 29 x 237 -> 输入 237 x 29 x 29
- 37 x 37 x 271 -> 输入 271 x 37 x 37
- 37 x 37 x 291 -> 输入 291 x 37 x 37
- 37 x 37 x 296 -> 输入 296 x 37 x 37
- 49 x 49 x 361 -> 输入 361 x 49 x 49

数据不插值、不裁剪、不填充到统一尺寸。同一个 batch 只装入相同尺寸的样本，不同 batch 可以具有不同尺寸。ResNet18 最后使用 `AdaptiveAvgPool3d(1,1,1)`，因此五种尺寸都能直接输入。

## 5. 归一化与增强

训练和验证都对每个完整三维样本单独计算 P1、P99：

`x = clip((x - P1) / (P99 - P1 + 1e-6), 0, 1)`

归一化范围是当前完整钻孔，不使用其他样本、验证集或外部板统计量，因此没有训练/验证统计量泄漏。归一化配置会写入每个模型检查点。

仅训练集使用原实验增强：高宽方向随机翻转、90 度旋转、平移，以及小幅噪声、亮度和对比度变化。长度方向不翻转。验证集不使用随机增强。

## 6. 模型与训练参数

- BasicBlock 数量：`[2, 2, 2, 2]`
- 通道：`64 -> 128 -> 256 -> 512`
- 参数量：33,143,106
- Dropout：0.5
- 损失：未加权 `CrossEntropyLoss`
- 优化器：AdamW
- 初始学习率：`1e-4`
- 权重衰减：`1e-3`
- 调度器：CosineAnnealingLR，`eta_min=1e-7`
- 训练轮数：每折 50
- batch size：4，同尺寸组 batch
- AMP：CUDA 可用时开启
- 五折训练种子：`42, 123, 2026, 3407, 777`
- 缺陷固定为正类 1，分类阈值固定为 0.5

## 7. 运行方法

完整运行：

```powershell
& "E:\pythonproject\3d_cnn_classify _K\train_val\run_resnet18_multiboard_clear_5fold.ps1"
```

默认数据位置为项目内的 `datasets\clear_binary_dataset`，无需传数据路径。也可以明确指定：

```powershell
& "E:\pythonproject\3d_cnn_classify _K\train_val\run_resnet18_multiboard_clear_5fold.ps1" -DatasetRoot ".\datasets\clear_binary_dataset"
```

若另一台电脑的 PyTorch 环境位置不同，可再传入 `-PythonExe "该环境的python.exe完整路径"`。

只训练一折用于先观察显存和速度：

```powershell
& "E:\pythonproject\3d_cnn_classify _K\train_val\run_resnet18_multiboard_clear_5fold.ps1" -Folds 0
```

如果显存不足，可加 `-BatchSize 2`。已经存在检查点时脚本会停止，避免误覆盖；明确重跑时才加 `-AllowOverwrite`。

也可以分别运行：

```powershell
python data_operate\make_multiboard_clear_5fold.py
python train_val\main_resnet18_multiboard_clear_5fold.py
```

## 8. 保存结果

模型保存在：

`model_best_last/resnet18_multiboard_clear_5fold/fold_0` 至 `fold_4`

每折保存：

- `best_f1_resnet18_3d.pth`
- `best_loss_resnet18_3d.pth`
- `last_resnet18_3d.pth`

主结果只按 `best_loss` 解释，另外两种检查点保留用于后续核对和外部测试。

训练验证结果保存在：

`train_val_result/resnet18_multiboard_clear_5fold`

主要文件：

- `fivefold_mean_std.csv/xlsx`：三个检查点的五折均值 ± 标准差，不含 Balanced Accuracy。
- `fivefold_checkpoint_metrics.csv/xlsx`：每折原始指标。
- `oof_predictions_best_loss.csv`：每个样本恰好一条 best-loss OOF 预测。
- `oof_metrics_best_loss.csv/json`：4403 个 OOF 样本合并后的总指标。
- 各折目录：训练历史、预测明细、混淆矩阵、ROC 和 PR 曲线。

Accuracy、Precision、Recall、F1、Specificity、AUC 和 AP 均以缺陷为正类计算。变量 batch 的 loss 按样本数加权求均值，不按 batch 简单平均。
