# Head40-733 模糊样本权重 1.25 实验说明

本实验在原版 Head40-733 上使用普通交叉熵，只提高模糊样本的训练权重，检验模型加强学习模糊样本后能否改善分类。分类始终是正常、缺陷两类。

## 实验配置

| 项目 | 设置 |
|---|---|
| 模型 | 原版 Head40-733，Stem 卷积核 7×3×3，池化核 3×3×3 |
| 训练数据 | 现有 Head40 多板数据及新版清晰/模糊清单 |
| 五折、seed | 沿用之前的五折划分和 seed |
| epoch / batch size | 50 / 4 |
| 优化器、学习率、增强、归一化 | 沿用之前配置 |
| 清晰正常、清晰缺陷训练权重 | 1.0 |
| 模糊正常、模糊缺陷训练权重 | 1.25 |
| 验证、测试损失 | 普通未加权 CE |
| 模型选择 | 未加权验证 CE 最小的 best_loss |
| 分类阈值 | 0.5 |

## 训练方法

先为每个样本计算普通 CE，再根据清晰/模糊标记设置样本权重。同一类中清晰与模糊样本可以具有不同权重，正常与缺陷使用相同规则，不额外设置类别权重。

```text
w_i = 1.0   （清晰样本）
w_i = 1.25  （模糊样本）
batch_loss = sum(w_i * CE_i) / sum(w_i)
```

该归一化方式与之前的样本加权实验一致。模糊标签的影响会增强，若其中存在错标，其影响也会增强，需要根据五折结果判断是否有收益。本实验不使用 Focal Loss 或标签平滑。

验证和测试对所有样本统一计算普通未加权 CE，指标按正常/缺陷计算。清晰/模糊只是附加分组信息，不会增加模型输出类别。

## 代码和数据准备

将以下三个新增文件复制到另一台电脑项目的对应目录：

```text
model/resnet18_3d_head40_toml_direction_unified_stem733_fuzzy_weight125_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem733_fuzzy_weight125_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem733_fuzzy_weight125_multiboard_3boards.py
```

加载器继续使用现有文件：

```text
data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py
```

模型、训练和测试是本实验独立的新文件。另一台电脑须已有原加载器、`multiboard_head40_toml` 训练数据和 `test_3boards_head40_toml` 测试数据，以及以下默认清单目录，其中包含清晰/模糊字段、五折划分和锁定三板测试清单：

```text
datasets/resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924
```

程序读取已经预先裁剪好的 Head40 RAW，用清单的 `difficulty` 字段区分清晰和模糊，加载时按原方式执行方向统一和归一化。若数据根目录位置不同，按原项目的路径配置调整，清单中的样本 ID 与五折划分应保持一致。

## 运行

在项目根目录运行五折训练：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem733_fuzzy_weight125_multiboard_5fold.py
```

训练结束后，测试各折的 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem733_fuzzy_weight125_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 保存位置和比较方式

```text
model_best_last/resnet18_head40_stem733_fuzzy_weight125_multiboard_5fold
train_val_result/resnet18_head40_stem733_fuzzy_weight125_multiboard_5fold
test_result/head40_stem733_fuzzy_weight125_3boards
```

各折模型文件名：

```text
best_loss_stem733_fuzzy_weight125.pth
best_f1_stem733_fuzzy_weight125.pth
last_stem733_fuzzy_weight125.pth
```

结果文件的保存格式沿用之前实验，包含每折指标、样本预测、曲线、OOF 及五折汇总。正式对比主看 `best_loss` 的五折算术均值，验证集和三板测试集分别与 Head40-733 普通 CE、Focal 实验比较。
