# Head40-533 Layer3 Branch E3 实验说明

## 1. 实验目的

本实验在 E2 的单尺度 D 卷积基础上改为并行 `k=3` 和 `k=5` 两个尺度，用于判断短距离与中距离轴向模式的联合建模是否带来稳定收益。

## 2. 模型结构

```text
                         Layer3 [B,256,10,H3,W3]
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
                  Layer4                       H/W Avg
                     │                        [B,256,10]
                     ▼                             │
                    GAP                    Conv1D k=1
                     │                         256→128
                M [B,512]                         │
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

两个 padding 分别为 1 和 2，因此两个尺度的输出深度都保持为 10，能够直接沿通道拼接。

## 3. 相对 E2 的改动

- 将单个 `Conv1D(k=3,128→128)` 替换为两个并行分支；
- `k=3` 分支输出 64 通道；
- `k=5` 分支输出 64 通道；
- 两个尺度拼接后仍为 128 通道；
- 后续 D-Max、640 维融合、Dropout(0.5) 和 FC 均保持不变。

因此 `E3 vs E2` 主要判断多尺度建模是否优于单尺度，而不是依靠扩大最终特征维度获得提升。

模型参数量为 `33,241,026`。

## 4. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e3_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E3_MULTIBOARD_README.md
```

## 5. 运行命令

检查模型结构：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_layer3_e3_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_5fold.py
```

训练完成后测试锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 6. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_layer3_e3_multiboard_22boards_5fold_locked_3boards
```
