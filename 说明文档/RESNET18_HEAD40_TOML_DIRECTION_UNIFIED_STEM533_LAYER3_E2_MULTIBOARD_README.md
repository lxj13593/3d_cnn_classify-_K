# Head40-533 Layer3 Branch E2 实验说明

## 1. 实验目的

本实验在 Layer3 Branch E1 的基础上增加单尺度 `Conv1D(k=3)`，用于判断相邻 D 位置之间的局部连续关系是否比逐位置通道投影更有效。

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
                                      Conv1D k=3, padding=1
                                               128→128
                                                  │
                                               BN+ReLU
                                                  │
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

`padding=1` 保证 `k=3` 前后 D 长度始终为 10。Layer4 仍走原始主干路径。

## 3. 相对 E1 的改动

- 在 `Conv1D(k=1)` 后增加 `Conv1D(128,128,kernel_size=3,padding=1)`；
- 增加对应的 `BatchNorm1d` 和 ReLU；
- 辅助向量、融合向量和分类器维度保持为 128、640 和 2；
- Dropout 继续固定为 `0.5`。

其余模型、数据和训练测试协议与 E1 完全一致。因此 `E2 vs E1` 主要回答单尺度 D 局部卷积是否有效。

模型参数量为 `33,224,642`。

## 4. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e2_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E2_MULTIBOARD_README.md
```

## 5. 运行命令

检查模型结构：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_layer3_e2_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_5fold.py
```

训练完成后测试锁定三板：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 6. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_22boards_5fold_locked_3boards
```
