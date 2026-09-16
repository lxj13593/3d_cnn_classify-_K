# Codex 任务迁移摘要

更新时间：2026-09-09

## 1. 用户偏好

- 使用中文沟通。
- 优先直接给结论和结果。
- 不做不必要、重复或耗时过长的检查。
- 展示实验结果时，不显示 Excel/CSV 文件名。
- 实验结果统一采用 `best_loss` checkpoint 的五折算术平均；验证指标和锁定测试指标分开列举。

## 2. 项目概况

工作区：`E:\pythonproject\3d_cnn_classify _K`

项目任务是使用 3D ResNet18 对钻孔三维体积进行正常/缺陷二分类：

- `normal = 0`
- `defective = 1`
- 输入排列为 `C×D×H×W`
- Head40 输入深度固定为 40
- 横截面支持 `29×29`、`37×37`、`49×49`
- Head40 根据 TOML 中的 `center2` 和 `bHeadUp` 生成
- 加载阶段统一 D 轴方向，使钻头端位于高 D 索引侧
- 每个样本使用自身 Head40 ROI 的 P1/P99 归一化
- 数据增强只作用于 H/W，不随机翻转 D

数据与评估协议：

- 训练/验证池：22 块板，共 4,364 个样本
- 正常样本 3,181，缺陷样本 1,183
- 锁定外部测试集：3 块板，共 778 个样本
- 测试集正常 556，缺陷 222
- 使用固定五折清单
- 折种子：42、123、2026、3407、777
- 训练 50 epoch，batch size 4
- AdamW，学习率 `1e-4`，weight decay `1e-3`
- CosineAnnealingLR，未加权 CrossEntropyLoss，AMP
- 保存 `best_loss`、`best_f1`、`last`
- 论文主比较使用 `best_loss`

## 3. Head40-533 基线

基线为 Head40-533 3D ResNet18：

```text
Stem Conv3D：5×3×3，stride=1
Layer1：64，stride=1
Layer2：128，首块 stride=2
Layer3：256，首块 stride=2
Layer4：512，首块 stride=2
AdaptiveAvgPool3d(1)
Dropout(0.5)
Linear(512→2)
```

已经核对实际模型和训练代码：基线确实带 `Dropout(0.5)`。

对于 Head40 输入：

- Layer3 输出深度为 `D=10`
- Layer4 输出深度为 `D=5`

## 4. 实验结果存放位置

已经完成并汇总的多板五折实验结果统一存放在：

```text
H:\3d钻孔缺陷分类实验结果_5折交叉验证_多板
```

其中：

```text
H:\3d钻孔缺陷分类实验结果_5折交叉验证_多板\train_val
```

用于存放各实验的五折训练与验证结果；

```text
H:\3d钻孔缺陷分类实验结果_5折交叉验证_多板\test
```

用于存放各实验在锁定三板上的测试结果。

此前列出的十组验证指标和测试指标，均从这个 H 盘总目录中读取，并统一采用 `best_loss` checkpoint 的五折算术平均。

新建的 Layer3 E1～E4 尚未正式训练。运行后默认在当前项目下分别生成：

```text
E:\pythonproject\3d_cnn_classify _K\model_best_last
E:\pythonproject\3d_cnn_classify _K\train_val_result
E:\pythonproject\3d_cnn_classify _K\test_result
```

其中每个 E1～E4 都有独立的实验子目录，不会覆盖基线或其他实验。若后续把新实验结果归档到 H 盘，应继续放入上述 H 盘总目录对应的 `train_val` 和 `test` 子目录。

## 5. 已完成实验的五折平均结果

以下均为 `best_loss` checkpoint 的五折算术平均。

### 5.1 验证指标

| 实验 | Loss | Acc(%) | Precision(%) | Recall(%) | F1(%) | Specificity(%) | AUC(%) | AP(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullVolume-733 | 0.167290 | 93.51 | 90.71 | 84.78 | 87.64 | 96.76 | 97.46 | 95.26 |
| Head40-333 | 0.153877 | 93.74 | 89.52 | 87.15 | 88.31 | 96.20 | 97.89 | 95.97 |
| Head40-533 | 0.151588 | 94.06 | 90.23 | 87.57 | 88.88 | 96.48 | 97.88 | 96.03 |
| Head40-733 | 0.152873 | 94.13 | 91.58 | 86.31 | 88.86 | 97.04 | 97.86 | 95.99 |
| Head40-733 + ACB | 0.154059 | 93.97 | 91.00 | 86.31 | 88.58 | 96.82 | 97.77 | 95.90 |
| Head40-533 + Layer2 DKeep | 0.158480 | 93.70 | 91.28 | 84.87 | 87.94 | 96.98 | 97.61 | 95.70 |
| Parallel Stem 333+511 | 0.153604 | 94.16 | 91.64 | 86.39 | 88.92 | 97.04 | 97.81 | 95.98 |
| Head40-533 + HWAvg-DMax | 0.154371 | 93.81 | 90.95 | 85.71 | 88.25 | 96.82 | 97.86 | 95.92 |
| Head40-533 + GAP+GMP | 0.159593 | 93.97 | 90.92 | 86.39 | 88.59 | 96.79 | 97.64 | 95.71 |
| Head40-533 + GAP+HWAvg-DMax | 0.152996 | 94.29 | 91.05 | 87.57 | 89.27 | 96.79 | 97.90 | 96.00 |

### 5.2 锁定三板测试指标

| 实验 | Loss | Acc(%) | Precision(%) | Recall(%) | F1(%) | Specificity(%) | AUC(%) | AP(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullVolume-733 | 0.178297 | 92.29 | 86.53 | 86.58 | 86.50 | 94.57 | 97.71 | 95.02 |
| Head40-333 | 0.167394 | 92.96 | 84.99 | 91.53 | 88.13 | 93.53 | 98.13 | 95.86 |
| Head40-533 | 0.160531 | 93.14 | 86.21 | 90.45 | 88.26 | 94.21 | 98.16 | 95.98 |
| Head40-733 | 0.160900 | 93.37 | 87.23 | 90.09 | 88.58 | 94.68 | 98.17 | 96.03 |
| Head40-733 + ACB | 0.159122 | 93.32 | 86.93 | 90.18 | 88.50 | 94.57 | 98.20 | 96.03 |
| Head40-533 + Layer2 DKeep | 0.172985 | 92.80 | 86.99 | 88.02 | 87.46 | 94.71 | 97.86 | 95.46 |
| Parallel Stem 333+511 | 0.166451 | 92.75 | 86.55 | 88.56 | 87.44 | 94.42 | 98.03 | 95.77 |
| Head40-533 + HWAvg-DMax | 0.161676 | 93.26 | 87.78 | 88.83 | 88.27 | 95.04 | 98.13 | 95.95 |
| Head40-533 + GAP+GMP | 0.162964 | 93.06 | 86.37 | 89.91 | 88.08 | 94.32 | 98.18 | 95.90 |
| Head40-533 + GAP+HWAvg-DMax | 0.167877 | 92.83 | 85.93 | 89.55 | 87.69 | 94.14 | 98.02 | 95.68 |

关键结论：

- 验证最低 loss：Head40-533，`0.151588`
- 验证最高 F1：Head40-533 + GAP+HWAvg-DMax，`89.27%`
- 测试最高 Accuracy/F1/AP：Head40-733，`93.37% / 88.58% / 96.03%`
- 测试最高 Recall：Head40-333，`91.53%`
- 测试最高 Precision/Specificity：Head40-533 + HWAvg-DMax，`87.78% / 95.04%`
- GAP+HWAvg-DMax 验证提升没有迁移到测试集，说明 D 分支存在过拟合风险

## 6. Layer3 渐进实验设计

共同原则：

- Layer3 输出后分为两条并行路径
- 主干仍为原始 `Layer3→Layer4→GAP`，输出 `[B,512]`
- 辅助分支从 Layer3 提取 D 方向信息，输出 `[B,128]`
- 最终拼接为 `[B,640]`
- 固定使用基线 `Dropout(0.5)` 和 `Linear(640→2)`
- 不同时加入 DKeep、ACB 或其他改动

### E1

```text
Layer3 [B,256,10,H,W]
→ H/W Average [B,256,10]
→ Conv1D k=1，256→128
→ BN + ReLU
→ D-Max [B,128]
→ 与 Layer4 GAP [B,512] 拼接
→ [B,640] → Dropout(0.5) → FC
```

目标：验证 Layer3 辅助信息是否有用；`k=1` 不混合相邻 D 位置。

### E2

```text
E1 的 k1 投影后
→ Conv1D k=3，128→128，padding=1
→ BN + ReLU
→ D-Max
```

目标：验证单尺度 D 局部连续性。

### E3

```text
k1 投影 [B,128,10]
├→ Conv1D k=3，128→64，padding=1
└→ Conv1D k=5，128→64，padding=2
→ concat [B,128,10]
→ D-Max [B,128]
```

目标：验证 `k3+k5` 多尺度是否优于单独 `k3`。

### E4

```text
Layer3
├→ H/W Average [B,256,10]
└→ H/W Maximum [B,256,10]
→ concat [B,512,10]
→ Conv1D k=1，512→128
→ 并行 k3/k5，各输出64
→ concat [B,128,10]
→ D-Max [B,128]
→ 与 Layer4 GAP 拼接
```

目标：验证 H/W Max 是否能补充 H/W Average。E4 是完整目标结构。

## 7. 已创建代码

已经新增四套完全独立的模型、训练、测试和说明，共 16 个文件。

### E1

- `model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e1_multiboard.py`
- `train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_5fold.py`
- `test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_3boards.py`
- `RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E1_MULTIBOARD_README.md`

### E2

- `model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e2_multiboard.py`
- `train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_5fold.py`
- `test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_3boards.py`
- `RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E2_MULTIBOARD_README.md`

### E3

- `model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e3_multiboard.py`
- `train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_5fold.py`
- `test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_3boards.py`
- `RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E3_MULTIBOARD_README.md`

### E4

- `model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e4_multiboard.py`
- `train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_5fold.py`
- `test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_3boards.py`
- `RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E4_MULTIBOARD_README.md`

四个实验分别使用独立的：

- 实验名
- checkpoint 文件名
- `model_best_last` 目录
- `train_val_result` 目录
- `test_result` 目录

没有启动正式五折训练。

## 8. 已完成的必要验证

- 12 个新增 Python 文件均通过 `py_compile`
- 四个模型均支持三种横截面尺寸
- Layer3/Layer4 输出形状正确
- 四个模型输出均为 `[B,2]`
- 四个模型的新增分支均可正常反向传播并获得有限梯度
- 四套训练与测试入口的实验名、模型目录和 checkpoint 名完全对应

## 9. 下一步建议

继续工作前先读取本文件。若用户准备运行实验，建议按 E1→E2→E3→E4 顺序训练；所有正式比较保持固定五折、相同训练参数和 `best_loss` 口径。

不要因为本摘要存在就重新生成上述 16 个文件；先检查工作区现状，仅在用户要求修改时再编辑。
