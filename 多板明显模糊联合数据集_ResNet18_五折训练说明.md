# 多板明显与模糊数据联合训练说明

## 1. 实验目的

本实验将明显数据集与模糊数据集合并，使用原始3D ResNet18进行二分类五折交叉验证。实验只改变训练数据，不改变模型结构、归一化、数据增强、损失函数和训练超参数，便于与之前的明显数据实验直接比较。

## 2. 联合数据规模

| 数据来源 | 正常 | 缺陷 | 合计 |
|---|---:|---:|---:|
| 明显数据集 | 3262 | 1141 | 4403 |
| 模糊数据集 | 1169 | 156 | 1325 |
| **联合数据集** | **4431** | **1297** | **5728** |

联合数据中正常与缺陷数量比约为 `3.42:1`。本实验继续使用不加权交叉熵，避免同时改变数据和损失函数。若后续缺陷Recall明显下降，再单独设置类别加权对照实验。

模糊数据中另有5张正常二维图因为一个孔序号对应多个RAW而未纳入，详细记录在 `fuzzy_binary_dataset/selection_audit.csv` 中。

## 3. 五折划分

- 共5折，划分随机种子为 `42`。
- 按“板号 + 正常/缺陷类别 + 明显/模糊难度 + RAW尺寸”联合分层。
- 每个样本恰好进入1折验证集，并进入其余4折训练集。
- 每折训练集约占80%，验证集约占20%。
- 每折训练集和验证集均包含全部15块板、明显数据和模糊数据。
- 训练集与验证集按 `sample_id` 检查，交集严格为0。
- 不设置同源内部测试集；后续新板数据作为真正的外部测试集。

已生成的划分清单位于：

```text
datasets/multiboard_clear_fuzzy_5fold/
```

清单只保存相对于数据父目录的路径，不复制RAW，也不写死盘符。数据父目录需要同时包含：

```text
clear_binary_dataset/
fuzzy_binary_dataset/
```

## 4. 模型与训练参数

| 项目 | 设置 |
|---|---|
| 模型 | 原始3D ResNet18 |
| BasicBlock数量 | `[2, 2, 2, 2]` |
| 通道数 | `64 -> 128 -> 256 -> 512` |
| 输入通道 | 1 |
| 分类类别 | 2 |
| Dropout | 0.5 |
| Epoch | 50 |
| Batch size | 4 |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-3` |
| 学习率调度 | CosineAnnealingLR |
| 最小学习率 | `1e-7` |
| 损失函数 | 不加权CrossEntropyLoss |
| 判定阈值 | 0.5 |
| 五折训练种子 | `42, 123, 2026, 3407, 777` |
| 主检查点 | `best_loss` |

不同尺寸样本不插值，使用同尺寸组批；训练 batch 每轮重新随机打乱。

## 5. 归一化与数据增强

每个RAW样本独立计算完整体积的P1和P99：

```text
x = clip((x - P1) / (P99 - P1), 0, 1)
```

训练集继续使用原数据增强：横截面翻转、横截面90度旋转、平移、噪声、亮度和对比度扰动。长度方向不翻转。验证集不使用任何随机增强。

## 6. 文件说明

```text
data_operate/make_multiboard_clear_fuzzy_5fold.py
    读取四类清单并生成联合五折划分。

data_operate/data_load_multiboard_clear_fuzzy.py
    联合数据加载、逐样本P1/P99归一化、训练增强和同尺寸组批。

train_val/main_resnet18_multiboard_clear_fuzzy_5fold.py
    五折训练、验证、检查点保存、曲线和OOF结果汇总。

train_val/run_resnet18_multiboard_clear_fuzzy_5fold.ps1
    自动寻找PyTorch环境，生成划分并启动训练。
```

## 7. 运行方式

在项目根目录运行：

```powershell
& ".\train_val\run_resnet18_multiboard_clear_fuzzy_5fold.ps1" `
  -DatasetParent "F:\325_275_and_sphere_data\钻孔数据325_275"
```

如果两个数据集已经放在项目的 `datasets` 目录中，可以省略 `-DatasetParent`：

```powershell
& ".\train_val\run_resnet18_multiboard_clear_fuzzy_5fold.ps1"
```

单独运行某一折：

```powershell
& ".\train_val\run_resnet18_multiboard_clear_fuzzy_5fold.ps1" `
  -Folds "0" `
  -DatasetParent "F:\325_275_and_sphere_data\钻孔数据325_275"
```

## 8. 输出结果

模型保存在：

```text
model_best_last/resnet18_multiboard_clear_fuzzy_5fold/fold_0...fold_4/
```

每折均保存：

- `best_f1_resnet18_3d.pth`
- `best_loss_resnet18_3d.pth`
- `last_resnet18_3d.pth`

训练和验证结果保存在：

```text
train_val_result/resnet18_multiboard_clear_fuzzy_5fold/
```

除五折均值、标准差、训练曲线、混淆矩阵、ROC、PR和逐样本预测外，还会生成：

- `oof_metrics_best_loss_by_difficulty.csv`：明显数据与模糊数据分别统计。
- `oof_metrics_best_loss_by_board.csv`：每块板分别统计。
- `oof_metrics_best_loss_by_board_difficulty.csv`：每块板的明显/模糊数据分别统计。

