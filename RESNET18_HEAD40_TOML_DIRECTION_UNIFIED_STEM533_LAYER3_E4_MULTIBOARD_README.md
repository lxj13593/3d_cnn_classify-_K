# Head40-533 Layer3 Branch E4 实验说明

## 1. 实验目的

本实验在 E3 双尺度轴向分支的基础上，将 Layer3 的 H/W 聚合从单独 Average 扩展为 Average 与 Maximum 拼接，用于判断局部强响应能否补充整体平均信息。

E4 是本组渐进实验的完整目标结构。

## 2. 模型结构

```text
                           Layer3 [B,256,10,H3,W3]
                                      │
                       ┌──────────────┴──────────────┐
                       │                             │
                       ▼                             ▼
                    Layer4                ┌──────────┴──────────┐
                       │                  │                     │
                       ▼                  ▼                     ▼
                      GAP             H/W Avg               H/W Max
                       │             [B,256,10]             [B,256,10]
                  M [B,512]                │                     │
                                          └──────────┬──────────┘
                                                     ▼
                                             Channel Concat
                                               [B,512,10]
                                                     │
                                             Conv1D k=1
                                                 512→128
                                                     │
                                                  BN+ReLU
                                                     │
                                               [B,128,10]
                                                     │
                                   ┌─────────────────┴─────────────────┐
                                   │                                   │
                                   ▼                                   ▼
                          Conv1D k=3, pad=1                    Conv1D k=5, pad=2
                                 128→64                              128→64
                                   │                                   │
                                BN+ReLU                            BN+ReLU
                                   │                                   │
                             [B,64,10]                           [B,64,10]
                                   └─────────────────┬─────────────────┘
                                                     ▼
                                             Channel Concat
                                               [B,128,10]
                                                     │
                                                   D-Max
                                                     │
                                                A [B,128]
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                               Concat [B,640]
                                      │
                               Dropout(0.5)
                                      │
                                Linear 640→2
```

Layer4 接收未经池化的原始 Layer3 特征；H/W Average 和 Maximum 只属于并行辅助分支。

## 3. 相对 E3 的改动

- 增加 Layer3 `H/W Maximum`；
- 将 `H/W Average` 与 `H/W Maximum` 按通道拼接为 `[B,512,10]`；
- `Conv1D(k=1)` 输入通道由 256 改为 512，输出仍为 128；
- 后续并行 `k=3/k=5`、D-Max、融合维度、Dropout 和 FC 均保持不变。

因此 `E4 vs E3` 主要判断双空间统计是否优于单独平均池化。

模型参数量为 `33,273,794`。

## 4. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e4_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E4_MULTIBOARD_README.md
```

## 5. 运行命令

检查模型结构：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_layer3_e4_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_5fold.py
```

训练完成后测试锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 6. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_layer3_e4_multiboard_22boards_5fold_locked_3boards
```
