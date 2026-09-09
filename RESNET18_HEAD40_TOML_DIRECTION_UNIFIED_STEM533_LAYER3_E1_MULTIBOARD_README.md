# Head40-533 Layer3 Branch E1 实验说明

## 1. 实验目的

本实验以 Head40-533 3D ResNet18 为基线，首次加入 Layer3 辅助分支，用于判断 Layer3 中保留的 `D=10` 轴向信息能否补充 Layer4 GAP 特征。

E1 不混合相邻 D 位置。`Conv1D(k=1)` 仅完成通道投影，作为后续 D 卷积实验的控制组。

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

Layer4 始终接收原始 Layer3 特征。辅助分支不会替代或修改 Layer4 主干。

## 3. 相对基线的改动

- 增加 Layer3 `H/W Average`，保留 D 维；
- 增加 `Conv1D(256,128,kernel_size=1)`、`BatchNorm1d` 和 ReLU；
- 对 D 维执行最大池化，得到 128 维辅助向量；
- 将分类器输入从 512 改为 `512+128=640`；
- Dropout 位置和概率保持基线的 `0.5`。

其余 Stem533、四个残差 stage、数据加载、方向统一、归一化、增强、固定五折、优化器、训练轮次和检查点规则均保持不变。

模型参数量为 `33,175,234`。

## 4. 独立文件

```text
model/resnet18_3d_head40_toml_direction_unified_stem533_layer3_e1_multiboard.py
train_val/main_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_5fold.py
test/test_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_3boards.py
RESNET18_HEAD40_TOML_DIRECTION_UNIFIED_STEM533_LAYER3_E1_MULTIBOARD_README.md
```

训练模型、检查点和测试结果使用独立实验名，不会覆盖基线或其他 Layer3 实验。五折清单与 Head40 数据加载器继续共享。

## 5. 运行命令

检查模型结构：

```powershell
python .\model\resnet18_3d_head40_toml_direction_unified_stem533_layer3_e1_multiboard.py
```

训练固定五折：

```powershell
python .\train_val\main_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_5fold.py
```

训练完成后测试锁定三板，正式比较使用 `best_loss`：

```powershell
python .\test\test_resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_3boards.py --checkpoint-kinds best_loss
```

## 6. 独立输出目录

```text
model_best_last/resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_22boards_5fold
train_val_result/resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_22boards_5fold
test_result/resnet18_head40_toml_direction_unified_stem533_layer3_e1_multiboard_22boards_5fold_locked_3boards
```

E1 首先与 Head40-533 基线比较，用于判断增加 Layer3 辅助信息是否值得继续。
