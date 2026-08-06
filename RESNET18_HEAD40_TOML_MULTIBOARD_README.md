# TOML Head40 多板 ResNet18 实验

## 数据

- 训练/验证集：`multiboard_head40_toml`，22 块板，4364 个样本。
- 独立测试集：`test_3boards_head40_toml`，3 块板，778 个样本。
- 文件序号 `N` 对应 TOML 的 `drillOutputSeq.(N-1)`。
- `bHeadUp=false`：`[center-15, center+25)`。
- `bHeadUp=true`：`[center-25, center+15)`。
- 所有输入长度均为 40；反向孔不进行方向翻转。

## 五折划分

Head40 根据 `SampleID` 严格继承完整体积基线的五折归属。每折约 80% 训练、20% 验证，不设置内部测试集。三块独立测试板不参与训练、验证、阈值选择或模型选择。

## 模型和参数

- 3D ResNet18，BasicBlock 数量为 `[2, 2, 2, 2]`。
- 通道数为 `64 -> 128 -> 256 -> 512`。
- stem 卷积和 maxpool 的步长均为 `(1, 1, 1)`，避免过早压缩 Head40。
- layer2、layer3、layer4 的步长为 2。
- 最终特征图：`40x29x29 -> 5x4x4`、`40x37x37 -> 5x5x5`、`40x49x49 -> 5x7x7`。
- AdaptiveAvgPool3d 输出 `1x1x1`，Dropout 为 0.5，分类数为 2。

## 训练设置

- 五折随机种子：`42, 123, 2026, 3407, 777`。
- Epoch：50；Batch size：4；初始学习率：`1e-4`。
- AdamW，weight decay 为 `1e-3`。
- CosineAnnealingLR，最低学习率为 `1e-7`。
- 未加权 CrossEntropyLoss。
- 每个 Head40 样本独立计算 P1/P99，截断后归一化到 `[0, 1]`。
- 训练增强与完整体积基线相同，只作用于横截面和灰度，不进行长度方向翻转。

## 运行

生成五折并训练：

```powershell
python .\data_operate\make_resnet18_head40_toml_multiboard_5fold.py
python .\train_val\main_resnet18_head40_toml_multiboard_5fold.py
```

也可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_head40_toml_multiboard_5fold.ps1
```

五折训练完成后进行三板独立测试：

```powershell
python .\test\test_resnet18_head40_toml_multiboard_3boards.py
```

测试会分别评估 `best_f1`、`best_loss` 和 `last`，其中 `best_loss` 为主要结果。
