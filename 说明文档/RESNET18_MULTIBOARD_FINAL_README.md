# 原始 ResNet18 多板五折训练与独立测试

## 数据范围

- 训练/验证：22 块板，正常 3181、缺陷 1183，共 4364 个样本。
- 独立测试：3 块板，正常 556、缺陷 222，共 778 个样本。
- 五折只划分训练/验证数据，独立测试板不参与训练、验证、归一化参数统计或模型选择。
- 已检查训练集与独立测试集：板号、样本 ID、RAW SHA256 均无重叠。

脚本会按数据集目录名自动扫描当前盘符，也可以通过参数显式指定路径，因此移动硬盘盘符变化后无需修改代码。

## 训练

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_val\run_resnet18_multiboard_final_5fold.ps1
```

训练配置：原始 3D ResNet18、五折、50 epoch、batch size 4、AdamW、未加权交叉熵、逐样本完整体积 P1/P99 归一化。训练阶段只进行横截面翻转、旋转、平移及轻量灰度增强，不翻转钻孔长度方向。

模型保存到：

```text
model_best_last/resnet18_multiboard_final_22boards_5fold/
```

训练验证结果保存到：

```text
train_val_result/resnet18_multiboard_final_22boards_5fold/
```

每折分别保存 `best_f1`、`best_loss` 和 `last`，主要结果以 `best_loss` 为准。

## 独立测试

五折全部训练完成后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\test\run_resnet18_multiboard_final_3boards.ps1
```

测试阶段不做数据增强、插值、裁剪、重新训练或阈值调参，使用与训练一致的逐样本完整体积 P1/P99 归一化。三个检查点类型分别测试，不混写。

结果保存到：

```text
test_result/resnet18_multiboard_final_22boards_5fold_locked_3boards/
```

主要文件包括五折均值与标准差表、逐折分板指标、五折集成指标、逐样本预测，以及分板和总体的混淆矩阵、ROC、PR 图。
