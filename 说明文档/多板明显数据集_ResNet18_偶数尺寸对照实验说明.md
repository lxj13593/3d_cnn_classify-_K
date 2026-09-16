# 多板明显数据集 ResNet18 偶数尺寸对照实验

## 实验目的

检验原始奇数尺寸输入改为偶数尺寸后，3D ResNet18 的五折验证结果是否发生变化。该实验只改变输入尺寸奇偶处理，五折样本、模型结构、归一化、数据增强、损失函数和训练参数均与原多板实验一致。

## 偶数化规则

不修改数据盘中的 RAW 文件，由新加载器在内存中处理：

1. 先读取原始完整钻孔并根据原始体积计算逐样本 P1/P99。
2. 训练集先执行与原实验相同的数据增强，再完成归一化和强度增强。
3. 对仍为奇数的维度，仅在正方向末端补 1 个体素，填充值复制当前边界体素。
4. 不裁剪、不插值、不改变任何原始体素，也不在头部起点前补体素。

尺寸对应关系：

| 原模型输入 D×H×W | 偶数实验输入 D×H×W | 最后特征图 |
|---|---|---|
| 237×29×29 | 238×30×30 | 8×4×4 |
| 271×37×37 | 272×38×38 | 9×5×5 |
| 291×37×37 | 292×38×38 | 10×5×5 |
| 296×37×37 | 296×38×38 | 10×5×5 |
| 361×49×49 | 362×50×50 | 12×7×7 |

偶数化不会扩大最后特征图。本实验比较的是输入边界和下采样对齐差异。

## 配套文件

- `data_operate/data_load_multiboard_clear_even.py`：独立偶数尺寸加载器。
- `model/resnet18_3d_multiboard_even.py`：独立保存的原始3D ResNet18，结构与原实验完全一致。
- `train_val/main_resnet18_multiboard_clear_even_5fold.py`：偶数尺寸五折训练验证脚本。
- `train_val/run_resnet18_multiboard_clear_even_5fold.ps1`：偶数尺寸实验启动脚本。

两套实验共同使用：

`datasets/multiboard_clear_5fold/fold_0` 至 `fold_4`

因此每个fold的训练和验证样本完全相同，可以直接进行配对比较。

两套加载器默认共同读取项目内的原始数据：

`datasets/clear_binary_dataset`

## 运行方法

当前原尺寸实验正在训练时不要同时启动本实验。原实验结束后运行：

```powershell
& ".\train_val\run_resnet18_multiboard_clear_even_5fold.ps1"
```

默认无需传数据路径，也可以明确指定项目内数据目录：

```powershell
& ".\train_val\run_resnet18_multiboard_clear_even_5fold.ps1" -DatasetRoot ".\datasets\clear_binary_dataset"
```

直接运行Python训练脚本也可以：

```powershell
python train_val\main_resnet18_multiboard_clear_even_5fold.py
```

## 独立输出

模型保存到：

`model_best_last/resnet18_multiboard_clear_even_5fold`

训练验证结果保存到：

`train_val_result/resnet18_multiboard_clear_even_5fold`

保存 `best_f1`、`best_loss`、`last` 三种模型以及训练曲线、预测明细、混淆矩阵、ROC、PR、五折均值标准差和 best-loss OOF 结果。主要仍比较 `best_loss`。
