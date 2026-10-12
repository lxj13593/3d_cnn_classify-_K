# Head40-733 + Layer2 3D ELA + CE 实验说明

## 实验目的与唯一变化

在已经完成的原始 Head40-733 + 普通 CE 对照上，只在 Layer2 整个阶段输出后、进入 Layer3 前，插入一个 3D ELA-S。不是在每个残差块中插入，也不叠加其他注意力。

原版 ELA 针对二维特征。本实验保留其轴向平均池化、Conv1D、GroupNorm、Sigmoid 和乘法加权结构，并扩展到 D/H/W 三个空间方向，属于三维适配，不声称原论文已经验证过本数据。

## 模型与注意力

主干保持原733：Stem Conv3D=7x3x3，stride1；原3x3x3池化、stride1；Layer1～4各两个残差块，通道64/128/256/512，阶段步长1/2/2/2；GAP、Dropout(0.5)、Linear(512→2)。

ELA输入为 [B,128,D,H,W]：

1. 对H/W平均，得到D方向 [B,128,D]；对D/W平均，得到H方向；对D/H平均，得到W方向。
2. 三方向共用同一个Conv1D和GroupNorm编码器。Conv1D：128→128、kernel5、padding2、groups16、无bias；GroupNorm：16组。
3. 分别经过Sigmoid，得到三个位置权重，并广播回五维。
4. 输出为原特征乘D、H、W三方向权重。无通道压缩，无额外残差，无0.5权重下限，也不改其他层。

注意力不是钻孔分割，也不保证删除背景或定位凸起。它学习位置权重；有效性用固定协议的分类结果验证。

| 原始输入 D×H×W | Layer2/ELA D×H×W | Layer3 | Layer4 |
|---|---|---|---|
| 40×29×29 | 20×15×15 | 10×8×8 | 5×4×4 |
| 40×37×37 | 20×19×19 | 10×10×10 | 5×5×5 |
| 40×49×49 | 20×25×25 | 10×13×13 | 5×7×7 |

新增参数5,376，预计总参数33,148,482。主干先按原初始化规则初始化，再创建ELA，因此相同模型构造种子下，原有主干的初始权重与原733相同；不承诺训练过程随机数流或结果逐位相同。

## 数据与训练协议

- 沿用已有加载器 data_operate/data_load_resnet18_head40_toml_direction_unified_multiboard.py，不重写、不修改。
- 原始Head40尺寸不变。不做55%裁剪、不去噪、不做背景扣除、不做Canny、不新建数据集。
- TOML方向统一仍在加载阶段完成：头部在高D侧；每个样本用自身完整Head40的P1/P99归一化。增强仍只作用于H/W，不随机翻转D。
- 沿用固定清单 datasets/resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924，不重新划分。22板4,364个训练/验证样本；锁定3板778个测试样本。
- 正常=0、缺陷=1。清晰/模糊不增加类别，全部训练样本损失权重1.0；无类别权重、模糊降权、Focal Loss或标签平滑。
- 从头训练：五折种子42、123、2026、3407、777；50 epoch、batch size4；AdamW，lr1e-4、weight decay1e-3；CosineAnnealingLR，eta_min1e-7；CUDA默认AMP。
- 训练、验证、测试均普通未加权CE。best_loss仍按验证CE最低选择；best_f1及last照旧保存。阈值固定0.5，测试不用于选模型或调参数。
- 对照为原始 Head40-733 + CE，不是533，也不是55%裁剪组。验证与外部测试分开汇报，只用best_loss五折算术均值，不展示集成。

## 拷贝与运行

把三个新代码文件拷到另一台电脑项目的同名子目录即可，不覆盖旧实验：

1. model/resnet18_3d_head40_stem733_ela_l2.py
2. train_val/main_resnet18_head40_stem733_ela_l2_ce_5fold.py
3. test/test_resnet18_head40_stem733_ela_l2_ce_3boards.py

原加载器、已有Head40数据、固定五折清单及已安装的PyTorch/NumPy/sklearn/matplotlib/tqdm等环境仍需在那台电脑存在。Excel输出需pandas/openpyxl；未安装时原脚本会提示，CSV仍保存。无需运行裁剪适配文件。

先运行训练，完成五折后另行运行测试：

```powershell
python train_val/main_resnet18_head40_stem733_ela_l2_ce_5fold.py
python test/test_resnet18_head40_stem733_ela_l2_ce_3boards.py
```

数据路径发现沿用旧加载器。如果未找到数据，给训练传 --dataset-root，给测试传 --test-dataset-root；清单路径不同则给两者传 --split-root，指向同一套既有清单。不要为此重新生成划分。

默认输出实验名 resnet18_head40_stem733_ela_l2_ce_5fold。输出路径使用短实验名和短板号，训练保存采用文件句柄，清单JSON读取兼容UTF-8 BOM。已有同实验checkpoint会拒绝覆盖，只有明确重跑时才使用 --allow-overwrite。

## 完整保存结果

```text
model_best_last/resnet18_head40_stem733_ela_l2_ce_5fold/fold_0～4/
  best_loss_ela_l2_ce.pth / best_f1_ela_l2_ce.pth / last_ela_l2_ce.pth

train_val_result/resnet18_head40_stem733_ela_l2_ce_5fold/
  experiment_config.json
  fold_0～4/运行配置、数据统计、训练历史和曲线
  fold_0～4/三个checkpoint的验证预测、指标、混淆矩阵、ROC、PR
  fivefold_checkpoint_metrics.csv / .xlsx
  fivefold_mean_std.csv / .xlsx
  best_loss OOF预测及总体、逐板、逐板逐尺寸诊断

test_result/resnet18_head40_stem733_ela_l2_ce_5fold_locked_3boards/
  predictions_all_folds.csv / fold_metrics_by_board.csv
  fivefold_mean_std.csv / .xlsx / fivefold_mean_std_raw.csv
  ensemble_predictions.csv / ensemble_metrics_by_board.csv
  test_config.json
  plots/三种checkpoint/fold_0～4/短板号/混淆矩阵、ROC、PR
  plots/三种checkpoint/短板号/原集成诊断曲线
```

默认完整测试五折的best_f1、best_loss、last。集成诊断仍保存，但不用于主比较。主要文件名、CSV列、Excel表和checkpoint结构仿照原完整保存版本，不删减旧输出。

checkpoint包含完整model_state_dict、优化器/调度器/AMP状态、原数据配置及ELA三维结构元数据。不是独立TorchScript模型；直接测试需使用本套模型与测试入口。测试严格核对架构、ELA配置和归一化元数据，不能混用原733、裁剪模型或其他注意力权重。

代码完成后仅做语法、合成输入形状/梯度与保存加载检查；不在本机启动正式训练或真实外部测试。

## 方法来源

Wei Xu, Yi Wan. ELA: Efficient Local Attention for Deep Convolutional Neural Networks. 本实验借鉴ELA-S并扩展到三维。原文：https://arxiv.org/abs/2403.01123。
