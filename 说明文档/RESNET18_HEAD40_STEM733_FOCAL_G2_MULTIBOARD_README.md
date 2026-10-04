# Head40-733 Focal Loss 实验说明

## 固定设置

```text
模型：原版 Head40-733
数据：当前清晰/模糊清单，不改
五折划分和 seed：不改
epoch：50
batch size：4
优化器、学习率、数据增强、归一化、方向统一：全部不改
分类阈值：0.5，不改
```

## 唯一改动

训练损失由普通交叉熵改为不带类别权重的 Focal Loss：

```text
loss = -(1 - p_t)^gamma * log(p_t)
gamma = 2.0
alpha = None
```

正常与缺陷不设置类别权重，清晰与模糊样本也不设置额外权重，所有样本的基础权重均为 `1.0`。

## 验证和模型选择

验证集仍使用普通、未加权的交叉熵和标准二分类指标。`best_loss` 仍按照未加权验证集交叉熵保存，保证和此前实验使用相同的比较口径。

锁定三板测试集也使用普通未加权交叉熵，分类阈值固定为 `0.5`。

## 文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem733_focal_g2_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem733_focal_g2_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem733_focal_g2_multiboard_3boards.py
```

数据加载继续使用现有的 Head40 direction-unified 加载文件，不重新生成数据或清单。

## 运行

五折训练：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem733_focal_g2_multiboard_5fold.py
```

训练完成后测试 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem733_focal_g2_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 输出目录

```text
model_best_last/resnet18_head40_stem733_focal_g2_multiboard_5fold
train_val_result/resnet18_head40_stem733_focal_g2_multiboard_5fold
test_result/head40_stem733_focal_g2_3boards
```

正式比较只读取五折均值，不使用集成结果。
