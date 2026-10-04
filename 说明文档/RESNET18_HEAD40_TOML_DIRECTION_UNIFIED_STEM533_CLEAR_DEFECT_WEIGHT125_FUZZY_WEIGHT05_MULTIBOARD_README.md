# Head40-533：清晰缺陷 1.25、模糊样本 0.5 实验说明

## 实验目的

本实验只做一个小改动：在现有“模糊样本权重 0.5”方案上，把**清晰缺陷样本**的训练权重从 `1.0` 提高到 `1.25`，观察缺陷召回率能否恢复，同时尽量控制精确率下降。

模型仍然只预测两类：

```text
normal = 0
defective = 1
```

`clear` 和 `fuzzy` 只是训练样本的可靠性标记，不是新的预测类别。

## 四组训练权重

| 样本组 | 权重 |
|---|---:|
| 清晰正常 | 1.0 |
| 清晰缺陷 | 1.25 |
| 模糊正常 | 0.5 |
| 模糊缺陷 | 0.5 |

每个样本先计算普通交叉熵，再按上表加权：

```text
loss_i = CE(logits_i, label_i)
loss = sum(weight_i * loss_i) / sum(weight_i)
```

没有使用 Focal Loss，也没有调整分类阈值。验证集和锁定三板测试集均使用普通未加权交叉熵，二分类阈值固定为 `0.5`。

## 独立代码文件

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard_clear_defect_weight125_fuzzy_weight05.py
model/resnet18_3d_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_3boards.py
```

模型结构与 Head40-533 保持一致，只有训练损失的样本权重发生变化。模型参数、优化器、调度器、五折划分、训练轮数、增强方式和方向统一规则均保持不变。

## 默认数据清单

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924
```

训练文件仍通过清单中的相对路径定位原始 `.raw` 数据，因此换电脑后要同时具备对应的数据集和这套清单。

## 运行方法

五折训练：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_5fold.py
```

训练完成后，使用 `best_loss` 模型测试锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 结果保存位置

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard_22boards_5fold
test_result/head40_stem533_cd125_fz05_locked_3boards
```

保存内容和此前 Head40-533 五折代码一致，包括每折的 `best_loss`、`best_f1`、`last` 模型，训练历史、验证预测、混淆矩阵、五折均值与 OOF 结果。正式对比时只看五折模型的算术均值，不使用集成结果。

为避免 Windows 路径长度限制，三个模型文件使用较短文件名：

```text
best_loss_stem533_cd125_fz05.pth
best_f1_stem533_cd125_fz05.pth
last_stem533_cd125_fz05.pth
```

测试结果目录也使用短名称，避免保存完整板号下的混淆矩阵时超过 Windows 路径长度限制。
