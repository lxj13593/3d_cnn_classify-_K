# Head40-533 IDO-GS 最终实验方案

版本日期：2026-09-21。后续实现与实验以本文为准；旧方案保留作历史记录。

## 1. 实验目的与执行范围

本次检验：训练过程中的预测错误统计，能否在不明显削弱模糊缺陷监督的前提下，改善 Head40-533 的分类效果。

实验名称为 IDO-GS，定位为 adapted IDO。第一阶段进行普通 CE 训练并检查统计信号；满足预先确定的条件后，第二阶段比较 IDO-GS 和公平的 CE 续训对照。

统计分组稳定不等于标签正确性已经得到证明。高错误率可能来自错标，也可能来自真实难缺陷、学习较慢或模型偏差。实验失败只表示本次协议未通过，不据此宣称 IDO 对所有设置都无效。

保持现有模型结构、数据及五折划分。按用户最新要求，数据加载、模型、训练和诊断全部独立新写，不导入旧实验代码，不占用 E1～E4 的文件和结果目录。目前先实现第一阶段，训练在另一台电脑执行。

## 2. 数据与历史参考

| 数据 | 总数 | normal | defect |
| --- | ---: | ---: | ---: |
| 22 板训练/验证池 | 4364 | 3181 | 1183 |
| 固定三板外部测试 | 778 | 556 | 222 |

4364 个样本是五折训练/验证池总量，并非某一折的验证量。每折使用原有 manifest；模型拟合、wrong event、BMM 和训练监督风险检查都仅使用该折训练样本。

历史结果如下，百分数已经四舍五入：

| 方案 | 五折 F1 均值 | 五折 AP 均值 | OOF 模糊缺陷检出数 / 总数 |
| --- | ---: | ---: | ---: |
| 模糊权重 1.0 | 88.88% | 96.03% | 79 / 144 |
| 模糊权重 0.5 | 88.67% | 95.77% | 75 / 144 |
| 模糊权重 0.0 | 86.27% | 93.90% | 43 / 144 |
| RACW | 88.83% | 95.94% | 67 / 144 |

这些结果提示削弱模糊缺陷监督可能增加漏检；它们本身不能证明所有模糊标签都正确。此次保留原标签，不删除或自动纠正样本。

主要新对照为同起点、同预算的 CE-continue；同一次 Stage 1 的 best_loss 则用于检查新方法是否比普通 CE 更好。历史结果另列参考，不用四舍五入后的数值代替原始指标作精细比较。

## 3. 固定设置与独立实现

| 设置 | Stage 1 | Stage 2 两分支 |
| --- | --- | --- |
| 模型 | 现有 Head40-533 | 加载该折 Stage 1 epoch-50 权重 |
| epoch | 50，完整执行 | 20 |
| 每 batch 原始样本数 | 4 | 4，每个样本两视图 |
| 优化器 | AdamW | 重新初始化 AdamW |
| 初始学习率 | 1e-4 | 1e-5 |
| weight decay | 1e-3 | 1e-3 |
| betas | (0.9, 0.999) | (0.9, 0.999) |
| scheduler | CosineAnnealingLR，T_max=50 | 新建 CosineAnnealingLR，T_max=20 |
| eta_min | 1e-7 | 1e-7 |
| 分类规则 | P(defect) >= 0.5 判 defect | 相同 |
| 模型选择 | 无权重验证 CE 最小 | 相同 |

fold 0～4 的随机种子固定为 42、123、2026、3407、777。模型的 dropout、归一化层、输入处理、ShapeBatchSampler 和其余参数沿用原始基线。

第一阶段独立代码：

- 模型：model/resnet18_3d_head40_stem533_ido_gs.py。
- 加载器：data_operate/data_load_head40_ido_gs.py。
- 训练入口：train_val/main_head40_ido_gs_stage1_5fold.py。
- BMM 诊断：train_val/ido_gs_stage1_diagnostics.py。

新加载器保留原数据根目录发现规则、manifest 字段和预处理语义。Stage 1 全样本使用普通、未加权 CE；不从人工模糊标记生成训练权重。

新 Dataset 默认单视图，并提供 num_views=2：从同一原始体积独立生成两次增强，保留原处理顺序和样本标识。第一阶段仅用单视图；确定性评估使用该折训练 manifest 对应的 augment=False 实例。

现有增强保留：H/W 翻转、HW 平面直角旋转、HW 各最多 3 体素平移、高斯噪声、亮度与对比度扰动。各增强概率及幅度沿用基线。保留 bHeadUp 方向校正，不增加随机 D 轴反转。几何增强在百分位归一化前、强度增强在后，不能用对归一化后 batch 再增强来冒充同一流程。

人工分组只用于训练监督风险检查和结果报告，不作为模型输入、单样本权重公式或采样条件。因此方法使用了分组信息进行实验准入，不能宣称整个实验完全不使用人工分组。

## 4. Stage 1：训练与错误统计

每折普通 CE 训练 50 轮，保存 best_loss 和 epoch-50。第 1～5 轮不累计错误；第 6～50 轮，每轮训练结束后，对该折训练集进行一次独立、无增强评估：

- model.eval()、no_grad；不更新参数或 BatchNorm 统计。
- 每个稳定 SampleID 恰好预测一次；记录 P(defect) 和按固定阈值得到的预测。
- 评估 loader 使用独立随机生成器；隔离 Python、NumPy、PyTorch 的训练随机状态。评估结束后恢复训练模式。
- 验证集、三板测试集均不进入此统计。

定义：

~~~text
wrong[t,i] = 1(pred[t,i] != given_label[i])
wrong_count[i] = sum(wrong[t,i], t=6,...,50)
wrong_rate[i] = wrong_count[i] / 45

PC[t] = mean_i(pred[t,i] != pred[t-1,i]), t=7,...,50
flip_rate[i] = mean_t(pred[t,i] != pred[t-1,i]), t=7,...,50
~~~

45 次观察对应 46 个可能错误率。错误次数相同并不意味着训练轨迹相同，例如“前期错、后期学会”和“反复预测变化”可能有相同计数。因此保存逐轮预测，flip_rate 只作辅助解释，不增加一个训练分支。

Stage 1 完成后才统一判断五折是否准入 Stage 2。若已有完全相同协议的完整逐轮记录与 checkpoint，可直接复用；只有 best_loss 或普通日志不能重建训练轨迹。

## 5. BMM：输入、拟合与结构门

每折对 given_label=normal 和 given_label=defect 分别拟合，输入为该折训练样本的 wrong_rate。单 Beta 和两 Beta 使用相同输入；先截断到 [1e-4, 1-1e-4]，不做随 epoch 改变的 min-max 归一化。

两分量按均值从低到高排序。低错误率分量称 clean，高错误率分量称 noise，仅作为模型内名称。输出 pc、pn、Fc、Fn，其中 pc+pn=1，Fc/Fn 是相应分量的 CDF。

数值实现固定如下：

| 项目 | 固定规则 |
| --- | --- |
| 精度 | CPU float64；密度、posterior 在 log 域计算 |
| 拟合目标 | 最大化 Beta 对数似然；两分量使用 EM，M 步为加权最大似然 |
| Beta 参数范围 | alpha、beta 均在 [1e-3, 1e3]；用 log 参数优化 |
| 混合初始权重 | (0.5, 0.5) |
| 两分量初值 | ((1,10),(10,1))、((1,4),(4,1))、((2,8),(8,2))、((2,5),(5,2))、((1,1),(2,1)) |
| 迭代 | EM 最多 200 次；平均对数似然相对变化 <=1e-6，连续 3 次后收敛 |
| 数值求解器 | 单 Beta 及 M 步使用有界 L-BFGS-B，最多 500 次；求解失败的初值作废 |
| 选择 | 从收敛、有限且不塌缩的初值中选择对数似然最高者 |

平均对数似然相对变化的分母为 max(1, 上次平均对数似然的绝对值)。若所有初值均无效，拟合失败。参数触及数值上下界、有效样本数不足或输入为常数时，记为退化拟合，不通过结构门。不得直接用加权矩估计替代 M 步最大似然后仍宣称执行了相同拟合。

~~~text
BIC = -2 * log_likelihood + k * log(n_class)
单 Beta：k=2
两 Beta：k=5
分量有效样本数 N_k = sum_i posterior[k,i]
~~~

每折、每个类别必须同时满足：

1. 拟合收敛，posterior/CDF 有限、取值合法，无退化分量。
2. BIC_2 < BIC_1 - 10。
3. 两个 N_k 均 >= max(0.05*n_class, 30)。
4. 高低分量均值差 >=0.10。

BIC 在此是离散错误率上的近似拟合筛选指标，不是标签噪声的显著性检验，也不要求混合密度一定有两个可见峰。

## 6. 稳定性检查与监督风险门

### 6.1 全窗口 Bootstrap：硬条件

对每折每个类别的训练样本做 100 次有放回重采样，每次样本数等于该类别原样本数。种子固定为 fold_seed+10000+class_id；保存实际索引或种子序列。

每次重拟合两 Beta，并在原始完整样本上计算 pn，与完整样本拟合的 pn 求 Pearson 相关。有效拟合需满足有限收敛、无数值边界退化、成分有效样本数下限和均值差下限；不重复比较单 Beta 的 BIC。

要求至少 90 次有效，且这些有效拟合的 posterior 相关系数中位数 >=0.90。相关系数因常数而无法定义的重采样计为无效，不丢弃后当作成功。

若该类完整窗口结构门已失败，跳过其 Bootstrap 并记录 skipped_due_to_structure_failure，准入结果仍为失败，避免进行不会改变结论的重复计算。

Bootstrap 仅检查当前样本池内的拟合稳定性，不将同板样本视为已证明相互独立，也不据此给泛化置信区间。

### 6.2 时间半窗和 PC：解释性诊断

前半窗为 epoch 6～27，共 22 次；后半窗为 epoch 28～50，共 23 次，各用自己的观察次数计算错误率。保存两半窗的拟合结果、与全窗口 posterior 的相关性；无法拟合时明确记录原因。

PC 的早期均值使用 t=7～10，后期均值使用 t=45～50，均包含两端。

半窗、PC 不作为额外硬条件。后半窗样本被学会、错误率集中到零，可能是正常训练结果；不能仅因后半窗单分量或相关性无法计算就判整个实验失败。NaN 或样本记录错误仍属于运行异常。

### 6.3 实际 CE 监督风险门：硬条件

IDO 中给定标签 CE 的系数为 pc。直接检查该折训练集 fuzzy-defect 的系数：

~~~text
R_fuzzy_defect = mean(pc | fuzzy-defect)
Q_fuzzy_defect = count(pc < 0.5 | fuzzy-defect) / n_fuzzy_defect

通过条件：R_fuzzy_defect >= 0.80 且 Q_fuzzy_defect <= 0.10
~~~

这两个阈值是本次预先固定的保守止损选择：平均 CE 系数至少保留 80%，低于一半监督的模糊缺陷最多占 10%。它们未被证明为最优值，也不保证召回率不会下降。

使用绝对监督保留率，避免 clear-defect 和 fuzzy-defect 同时被大幅降权时，相对比较仍然放行。clear-defect、reviewed_defect 等组的 pc 分布同时报告。

这里检查的是 CE 的系数，不是实际梯度大小。不要把 pc 截断到 0.8；该规则用于决定是否继续整个实验，不改变单样本损失。

可额外列出 pn>=0.5 且 epsilon>=缺陷类 epsilon 中位数的冲突样本，但这只是描述性分组。高 epsilon 不豁免监督风险门：两个视图同时预测正常时，一致性好并不意味着缺陷标签获得了有效监督。

### 6.4 五折统一准入

五折的两个类别均通过结构门和 Bootstrap，且五折各自均通过 CE 监督风险门，才进入 Stage 2。分组缺失或目标组为空时，标记数据协议不完整并停止，不能默认通过。

失败时输出具体 fold、类别、未通过项目和数值，本次实验到此结束。失败结论写“未通过本次准入条件”，不写“已证明没有标签噪声”。

## 7. Stage 2：公平分叉

从每折 Stage 1 epoch-50 checkpoint 复制模型参数及 BatchNorm buffers，分别进入 CE-continue 和 IDO-GS。使用 epoch-50 是预先固定的简化选择；不宣称从更早训练 checkpoint 分叉本身构成数据泄漏。

两分支重新初始化优化器、scheduler、GradScaler，均执行 20 轮。每折每轮两分支使用相同的 sampler/增强随机种子：fold_seed+20000+s，其中 s=1,...,20；同一样本的两个视图独立生成，分支间可复现相同抽样。

每个 batch 执行两次顺序 forward，然后合并损失执行一次 backward 和 optimizer.step。保持两分支的 forward 顺序和 BatchNorm 行为一致，不把其中一条分支的两视图拼成 batch=8。

CE-continue 同样保留确定性训练集评估记录，但这些记录不生成训练权重。常规验证均为单视图、无增强。

## 8. 唯一损失定义

设 logits z1、z2 的形状均为 [B,2]，y 为给定标签。两条分支的 CE 都使用 logits 作为输入、reduction=none，得到每样本值。

难度系数采用 IDO 论文公式（5）：

~~~text
epsilon = pc * Fc + pn * (1 - Fn)
~~~

不采用此前修订稿中的 (1-pn) 第二项。核对时官方 WELoss 源码与论文在此不一致，本文选择论文公式。[IDO 原论文，公式（5）](https://proceedings.neurips.cc/paper_files/paper/2025/file/429e7b31625a8b7839f9e4d6e2aa9bb9-Paper-Conference.pdf)，[官方 WELoss 源码](https://github.com/iTheresaApocalypse/IDO/blob/main/utils/lnl_methods.py)

逐样本定义：

~~~text
p1 = softmax(z1)
p2 = softmax(z2)
p_bar = (p1 + p2) / 2

CE_i = CE(z1_i, y_i) + CE(z2_i, y_i)
MSE_i = mean_over_2_classes((z1_i - z2_i)^2)
confidence_i = max_over_2_classes(p_bar_i)
H_i = -sum_over_2_classes(p_bar_i * log(clamp_min(p_bar_i, 1e-8)))

L_CE_continue = mean_over_batch(CE_i)

L_clean_i = pc_i * CE_i
L_hard_i = epsilon_i * MSE_i
L_noise_i = pn_i * confidence_i * H_i
L_IDO_GS = mean_over_batch(L_clean_i + L_hard_i + L_noise_i)
~~~

两分支均使用双视图 CE 之和，不在其中一条分支额外除以 2。MSE 明确作用于 logits，保留当前官方实现的这一选择。所有项系数固定为 1，不做系数扫描。

pc、pn、CDF 和 epsilon 不参与反向传播；p_bar、confidence、H 保留梯度。熵项为逐样本计算，三个向量均为 [B]，不能通过 [B,1] 与 [B] 广播得到 [B,B]。[官方实现参考](https://github.com/iTheresaApocalypse/IDO/blob/main/utils/lnl_methods.py)

该损失仍可能强化已有错误预测，因此“难度系数正确”不等于“真难缺陷自动得到保护”，必须执行监督风险门和最终召回约束。

## 9. Stage 2 动态更新与停止规则

Stage 2 第 1 轮使用 Stage 1 完整窗口的 BMM。之后每轮训练结束，用该分支当前模型无增强评估训练集，每个样本追加一次错误记录：

~~~text
第 s 轮训练使用：上一轮有效 posterior
第 s 轮训练后评估：n_observed = 45 + s
wrong_rate_s = cumulative_wrong_count / (45+s)
新 posterior 用于第 s+1 轮
~~~

IDO-GS 每轮更新 class-wise BMM：优先以上次参数初始化；如该初值无效，再尝试第 5 节固定初值，属于预定数值重试而非参数搜索。统计坐标始终使用相同错误率截断方式。

每轮仅检查：有限收敛、参数不触及数值边界、posterior/CDF 范围合法、两个分量有效样本数仍达到下限、均值差仍 >=0.10，并重算廉价的 CE 监督风险门。Stage 2 不重复 BIC、100 次 Bootstrap 或半窗拟合。

运行结论分别标注：

- fit_invalid：数值、分量退化或分离度条件失败。
- supervision_risk_stop：拟合有效，但 CE 监督风险门失败。
- runtime_error：代码、设备或数据读取错误，不能当作算法性能失败；修复后用原配置恢复。

任一 fold 出现前两种停止情况，整个五折 IDO-GS 本次不通过；保留已产生的记录，停止剩余 Stage 2 训练，不用其余成功折的均值冒充五折结果。不冻结旧 posterior，也不临时切换普通 CE 继续算作 IDO。

## 10. AMP、保存与必要验证

网络前向沿用 AMP。自定义损失在关闭 autocast 的区域内，将 z1、z2 转为 float32；权重和 CDF 也显式转为同设备的 float32，在此计算 softmax、CE、MSE 和熵，最后使用 GradScaler 反向传播。

不要将 logits.dtype 当作损失必须使用的 dtype；AMP 下 logits 可能为半精度，而敏感损失通常需要 FP32。[PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)

进入 loss 前核对权重形状 [B]、posterior 和为 1、所有值有限，避免过去 Float/Half 索引赋值错误和跨样本广播。

保存 epoch-50 起点与断点时同时保留模型、优化器、scheduler、GradScaler、随机状态、SampleID 顺序、wrong_count、观察次数和 BMM 参数。跨机器继续时不改变数据身份，路径可按既有配置映射。

后续实现只做与这次新增逻辑相关的必要检查：SampleID 一轮一次、公式的简单极限值、逐样本 shape、AMP 下 loss/梯度有限，以及同起点两分支的配置一致性。训练在用户的运行电脑执行。

## 11. 模型选择与成功标准

Stage 1 按现有规则保存最小无权重验证 CE 的 best_loss，完整训练到 50 轮。Stage 2 两分支分别从 epoch 1～20 中按同一规则选择 best_loss；loss 严格更小时更新，完全相同则保留较早 epoch。共同起点另行保留，不混入 Stage 2 候选。

所有验证指标使用该折验证集、单视图无增强和固定 0.5 阈值。模型选择不使用 BMM loss、模糊组指标或三板结果。

五折均完整有效后，以各折 best_loss 计算算术均值。以下条件必须全部满足：

| 项目 | 成功条件 |
| --- | --- |
| 主对照 F1 | IDO-GS 五折均值比 CE-continue 高至少 0.30 个百分点 |
| 折间方向 | 至少 3/5 折 IDO-GS F1 >= 同折 CE-continue |
| 普通 CE 参照 | IDO-GS 平均 F1 >= 同一次 Stage 1 best_loss 平均 F1 |
| AUC、AP | 分别相对 CE-continue 和同次 Stage 1 均不得下降超过 0.10 个百分点 |
| 模糊缺陷 OOF | TP 不少于 CE-continue、不少于同次 Stage 1，且至少 79/144 |
| 单折模糊缺陷 | 相比对应 CE-continue，Recall 下降必须小于 10 个百分点 |
| 运行有效性 | 五折完整，无 fit_invalid 或 supervision_risk_stop |

所有比较使用未四舍五入的指标；0.30 个百分点对应数值 0.003。79/144 使用整数计数判定，不使用展示用 54.86% 反推。

OOF 的 144 个模糊缺陷每个只计一次。OOF 子组召回用于检查漏检，不替代主要指标的五折均值。五折有重叠训练数据，3/5 折占优和均值差均不作为统计显著性的证明。

条件失败时写明失败项目，本次停止；不据此调整阈值再反复重跑。

## 12. 外部比较与结果交付

仅当第 11 节全部通过，才对固定三板分别评估 CE-continue 和 IDO-GS 的五个 best_loss 模型。每个 fold 各自测试后计算指标均值；不临时改成五模型概率集成。历史方案只引用已有相同口径结果。

三板共 778 个不同样本；五个模型重复评估不能写成有 3890 个独立测试样本。这三板已经用于过往方案比较，因此称“固定外部测试集”，不称从未使用过的最终盲测集。此次结果不反馈到训练配置。

固定阈值下若 IDO-GS 外部 F1 均值不高于 CE-continue，则写“五折条件通过，但外部优势未得到确认”；外部结果不触发下一轮调参。

只生成以下必要实验产物，统一保存在独立的 IDO-GS 结果目录：

| 产物 | 内容 |
| --- | --- |
| protocol.json | 本方案版本、代码版本、数据/划分身份、种子和固定参数 |
| stage1_epoch_predictions.csv | fold、SampleID、epoch、标签、预测、P(defect) |
| stage1_bmm_diagnostics.json | 类别拟合、BIC、Bootstrap、半窗、PC、准入结论与失败原因 |
| stage1_sample_diagnostics.csv | 样本分组、错误计数、flip_rate、pc/pn、CDF、epsilon |
| checkpoints | Stage 1 best_loss/epoch-50；两分支 best_loss/恢复断点 |
| stage2_history.csv | 每折每轮训练三项 loss、验证指标、BMM 状态和 CE 监督保留率 |
| stage2_sample_diagnostics.csv | IDO-GS 每轮 SampleID、错误计数、pc/pn、epsilon、预测 |
| comparison_folds.csv | 各折分别的主要指标与分组计数 |
| comparison_mean.csv | 五折主要指标的算术均值 |
| oof_predictions_best_loss.csv | 每个样本只出现一次的验证预测，按方法区分 |
| external_comparison.csv | 仅准入后生成的外部测试逐折结果和均值 |

主要指标报告 Loss、Acc、Precision、Recall、F1、Specificity、AUC、AP。分组分别报告：

- clear-normal、fuzzy-normal：样本数、正确数/TN、FP、正常正确率。
- clear-defect、fuzzy-defect：样本数、检出数/TP、FN、Recall。
- reviewed_defect、supplemental：按现有分组身份另列，使各互斥分组计数与全体对应；不擅自并入 clear。

单一真实类别子组不解释 F1/AUC。对比页主要显示五折均值；每折明细保留在结果文件中，按需列出。

执行顺序固定为：五折 Stage 1 → 统一准入 → 五折 CE-continue 与 IDO-GS 各 20 轮 → 五折和 OOF 判定 → 条件通过后一次固定三板比较。本次不增加模型结构、Teacher、自动改标签或额外实验分支。
