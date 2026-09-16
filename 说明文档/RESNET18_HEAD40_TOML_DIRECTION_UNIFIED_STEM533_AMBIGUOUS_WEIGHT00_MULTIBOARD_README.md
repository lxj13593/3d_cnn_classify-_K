# Head40-533 模糊样本权重 0.0（Clear-only）实验

本实验是权重 0.5 的后续消融：仍为二分类 `normal=0`、`defective=1`，只将
`difficulty == "fuzzy"` 的训练损失权重设为 `0.0`。

训练时，模糊样本会在模型前向计算前移除；若一个形状批次全部为模糊样本，则跳过该批次。
因此模糊样本不产生梯度，也不更新 BatchNorm 统计量。验证集和锁定三板测试集始终采用普通、
未加权的交叉熵与二分类指标，主检查点仍为 `best_loss`。

保持不变：Head40-533 网络结构、五折划分、随机种子、数据增强、优化器、学习率、50 epoch、
验证规则和锁定三板测试集。

独立文件：

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard_ambiguous_weight00.py
model/resnet18_3d_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_3boards.py
```

训练：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_5fold.py
```

训练完成后，以 `best_loss` 测试锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_3boards.py --checkpoint-kinds best_loss
```

输出目录与 0.5 实验完全分离：

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard_22boards_5fold_locked_3boards
```
