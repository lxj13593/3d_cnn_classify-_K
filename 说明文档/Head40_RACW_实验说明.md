# Head40-533 RACW 实验说明

## 1. 实验目的

在现有 Head40-533 二分类任务上，对 difficulty == "fuzzy" 的样本做动态损失加权。

~~~text
normal = 0
defective = 1
~~~

不修改模糊样本标签，不使用 Teacher、soft label、pseudo label 或标签纠正。该实验单独命名为 RACW，不覆盖已有 E1～E4 脚本、模型或结果。

## 2. 使用的文件

~~~text
模型：
model/resnet18_3d_head40_toml_direction_unified_stem533_racw_multiboard.py

五折训练：
train_val/main_resnet18_head40_toml_direction_unified_stem533_racw_multiboard_5fold.py

锁定三板测试：
test/test_resnet18_head40_toml_direction_unified_stem533_racw_multiboard_3boards.py
~~~

不新建数据加载文件。训练直接复用已有的多板 Head40 加载器、数据划分、归一化、方向统一和数据增强流程。

## 3. 固定实验设置

~~~text
网络：Head40-533 3D ResNet18
五折划分：沿用原来的 22 板固定划分与随机种子
训练轮数：50
batch size：4
优化器：AdamW
学习率：1e-4
权重衰减：1e-3
学习率调度：CosineAnnealingLR
主检查点：best_loss（未加权验证 Loss）
~~~

RACW 固定参数：

~~~text
Warm-up：前 5 个 epoch，全部样本普通 CE，权重均为 1.0
类中心来源：当前训练折、difficulty == "clear" 的样本
类中心特征：Layer4 GAP、dropout 前的 512 维特征
高可靠条件：预测类别等于原标签，且原标签概率 >= 0.8
最小中心样本数：clear-normal 与 clear-defective 各至少 10 个
中心更新：EMA，mu = 0.99
可靠度温度：tau = 0.2
模糊样本最低权重：0.3
~~~

从 epoch 6 开始：

~~~text
clear / reviewed_defect / supplemental：权重 = 1.0
fuzzy：权重 = 0.3 + 0.7 * sigmoid((s_y - s_other) / 0.2)
~~~

若第 5 个 epoch 后任一类别不足 10 个高可靠清晰样本，当前折会报错停止；不会使用零中心或伪造中心。

## 4. 运行训练

在项目根目录执行：

~~~powershell
python train_val\main_resnet18_head40_toml_direction_unified_stem533_racw_multiboard_5fold.py --device auto
~~~

如果新电脑不能自动找到训练数据集，再显式传入训练数据根目录：

~~~powershell
python train_val\main_resnet18_head40_toml_direction_unified_stem533_racw_multiboard_5fold.py --dataset-root "H:\多板数据集\multiboard_head40_toml" --device auto
~~~

默认输出：

~~~text
model_best_last/resnet18_head40_toml_direction_unified_stem533_racw_multiboard_22boards_5fold/
train_val_result/resnet18_head40_toml_direction_unified_stem533_racw_multiboard_22boards_5fold/
~~~

## 5. 五折验证与比较

主看 best_loss。关键输出：

~~~text
fivefold_checkpoint_metrics.csv
fivefold_mean_std.csv
oof_predictions_best_loss.csv
oof_group_metrics_best_loss.csv
fold_k/validation_group_metrics_best_loss.csv
fold_k/training_history.csv
~~~

分组文件只分别解释：

~~~text
clear-normal：正确率、FP
clear-defect：Recall、FN
fuzzy-normal：正确率、FP
fuzzy-defect：Recall、FN
~~~

不要对单一真实类别分组解释 F1。

RACW 只与已有固定模糊权重 1.0、0.5、0.0 对照比较。五折验证满足以下条件，才进行锁定三板测试：

~~~text
F1 不低于 weight = 1.0 基线
且 AUC、AP 不下降
且 fuzzy-defect Recall 不低于 weight = 1.0 基线
~~~

若不满足，停止 RACW 方向，不根据三板结果回调 p_thr、tau、w_min 或 mu。

## 6. 锁定三板测试

只有五折验证满足第 5 节条件后，执行：

~~~powershell
python test\test_resnet18_head40_toml_direction_unified_stem533_racw_multiboard_3boards.py --test-dataset-root "H:\多板数据集\test_3boards_head40_toml" --checkpoint-kinds best_loss --device auto
~~~

默认输出：

~~~text
test_result/resnet18_head40_toml_direction_unified_stem533_racw_multiboard_22boards_5fold_locked_3boards/
~~~

三板测试始终使用普通、未加权的二分类评价；RACW 只改变训练阶段的模糊样本损失权重。
