# TorchScript 模型测试说明

## 1. 模型文件

### 全体积模型

~~~text
model_best_last/best_loss_resnet18_3d_direction_unified_torchscript.pt
~~~

- 输入完整钻孔体积。
- 支持训练数据中的不同体积尺寸，例如 `237×29×29`、`296×37×37`、`361×49×49`。

### Head40 模糊样本权重 0.5 模型

~~~text
model_best_last/best_loss_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight05_torchscript.pt
~~~

- 输入已经按 TOML 截取的 Head40 数据，不能输入完整体积。
- 深度必须为 `40`，横截面支持 `29×29`、`37×37` 和 `49×49`。

## 2. 输入要求

- 输入张量形状为 `[B, 1, D, H, W]`。
- 数据类型为 `torch.float32`。
- 同一个 batch 内的样本尺寸必须一致；不同尺寸应分批测试。
- 不调整原始横截面尺寸，不进行插值或缩放。
- 按训练方式统一钻头方向：`bHeadUp=true` 时沿深度轴翻转，使钻头位于高深度索引侧。
- 每个样本单独计算第 1、99 百分位，截断后归一化到 `[0,1]`。全体积模型在完整体积上计算，Head40 模型在截取后的 Head40 上计算。

## 3. 加载与预测

~~~python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("模型文件.pt", map_location=device)
model.eval()

# volume 必须已经完成方向统一和归一化
# 形状：[B, 1, D, H, W]
volume = volume.to(device=device, dtype=torch.float32)

with torch.inference_mode():
    logits = model(volume)
    probability = torch.softmax(logits, dim=1)
    defect_probability = probability[:, 1]
    prediction = (defect_probability >= 0.5).long()
~~~

输出 `0` 表示正常，输出 `1` 表示缺陷。模型输出的是 `[B,2]` logits，必须先做 `softmax`；索引 1（第二列）是缺陷概率，固定阈值为 `0.5`。

## 4. 测试要求

- 测试时禁止数据增强，模型必须执行 `eval()`。
- 使用固定三板测试集时，所有样本都要测试，不能根据结果删样本或调整阈值。
- 五折模型应分别测试后计算五折指标均值；单个模型的结果不能写成五折结果。
- 当前转换的两个 `.pt` 都是 Fold 4 的 `best_loss` 模型，只能作为单折模型测试，不能代表完整五折结果。
- 保存 Accuracy、Precision、Recall、F1、Specificity、AUC 和 AP，并保留逐样本预测结果。

## 5. 注意事项

- 两个模型的输入数据不同，不能混用。
- `.pt` 已包含模型结构和权重，不需要再创建 ResNet18 后加载 `state_dict`。
- `.pt` 不包含数据读取和预处理，方向统一、Head40 截取和归一化仍需由测试代码完成。
