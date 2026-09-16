# Head40-533 模糊样本权重 0.5 实验

## 实验目的

本实验严格执行《模糊样本训练处理方案》的第一组对照：保持 Head40-533
模型、数据划分、增强、优化器、训练轮数、检查点选择和锁定测试集不变，
只在训练阶段降低模糊样本的监督权重。

模型输出始终为二分类：

```text
normal = 0
defective = 1
```

`fuzzy` 不是第三类，也不是新的预测目标。

## 模糊标记和训练损失

数据清单的 `difficulty` 用于产生训练批次中的 `ambiguous` 标记：

```text
difficulty == fuzzy  -> ambiguous = 1 -> weight = 0.5
其他 difficulty       -> ambiguous = 0 -> weight = 1.0
```

训练使用逐样本交叉熵：

```text
loss_i = CE(logits_i, label_i)
loss = sum(weight_i * loss_i) / sum(weight_i)
```

验证和锁定三板测试仍使用普通、未加权的交叉熵和正常二分类指标。`best_loss`
仍由未加权验证 Loss 选择，因此与 Head40-533 基线的主比较口径一致。

## 独立文件

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard_ambiguous_weight05.py
model/resnet18_3d_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_3boards.py
```

模型使用独立文件实现，结构与已有 Head40-533 完全一致；本实验不增加 Attention、
ECA、SE、CBAM、ACB、D 分支或新的池化模块。

## 运行

检查训练参数：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_5fold.py --help
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_5fold.py
```

完成训练后，以 `best_loss` 评估锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_3boards.py --checkpoint-kinds best_loss
```

若默认 `python` 没有 PyTorch，请使用实际的 PyTorch 环境解释器执行同一命令。

## 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_multiboard_22boards_5fold_locked_3boards
```

训练结果额外按 `difficulty` 汇总 best-loss OOF 指标，便于核对清晰和模糊样本的
差异；主比较仍使用全体样本的五折算术平均。
