# Head40-533 IDO 门控小样本实验方案（建议执行版）

## 结论先行

这不是一套默认要跑完的 IDO 训练，而是一项 **先证伪、后训练** 的实验。

RACW 已经显示：模糊缺陷中有相当一部分是“真难缺陷”，不是可随意降权的噪声。因而 IDO 只有在五折训练轨迹确实能稳定地区分“疑似错标”和“真难样本”时，才允许进入第二阶段。任何诊断门不通过，就停止这条路线，保留当前固定权重 1.0 基线。

实验编号使用 `IDO-GS`（Gated Small-sample），不占用、不覆盖 E1～E4。

## 0. 固定边界

- 沿用当前 Head40-533 模型、22 板五折划分、随机种子、评价方式和锁定 3 板测试集。
- 直接复用现有多板数据加载器；不新写、不修改数据加载逻辑。
- 训练时不读取“模糊/清晰”字段来影响损失、采样或预测；该字段仅用于训练后的保护性诊断和报告。
- 不加 Teacher、软标签、强增强、数据重采样或参数搜索。
- 每个阶段只允许一组预先写死的参数；不得根据某折结果临时改阈值。

当前必须守住的参考结果如下（均为已有结果，不重新生成）：

| 方案 | 五折 F1 均值 | 五折 AP 均值 | OOF 模糊缺陷召回 |
| --- | ---: | ---: | ---: |
| 固定模糊权重 1.0 | 88.88% | 96.03% | 79/144 = 54.86% |
| 固定模糊权重 0.5 | 88.67% | 95.77% | 75/144 = 52.08% |
| 固定模糊权重 0.0 | 86.27% | 93.90% | 43/144 = 29.86% |
| RACW 动态权重 | 88.83% | 95.94% | 67/144 = 46.53% |

所以，这个实验的重点不是再把模糊缺陷普遍减权，而是验证训练轨迹中是否存在稳定、可信的“疑似错误标签”信号。

## 1. 总流程

```text
五折 Stage 1：普通 CE，50 epoch
        ↓
wrong-event / Beta 混合 / 模糊缺陷保护门全部通过？
        ├─ 否：报告诊断并停止 IDO
        └─ 是：从相同 epoch-50 checkpoint 出发
                  ├─ IDO-GS Stage 2，20 epoch
                  └─ CE-continue 对照，20 epoch
                         ↓
                五折预注册成功条件通过？
                         ├─ 否：停止，不上锁定测试集
                         └─ 是：仅一次锁定 3 板外部测试
```

刻意不采用“跑完 50 轮后回头挑一个 Label-Wave 最优 checkpoint”。那种做法会用 checkpoint 之后的轨迹来选择 checkpoint，逻辑不干净。这里的 Stage 2 一律从 **epoch 50** 的同一 checkpoint 开始；Label-Wave 只作为 Stage 1 的稳定性诊断，不承担挑模型的职责。

## 2. Stage 1：普通 CE 训练与 wrong-event 采集

### 2.1 训练

五折分别执行当前固定权重 1.0 基线的原始 CE 训练 50 epoch。模型、优化器、学习率、batch、常规轻度增强、早停/保存方式均与基线一致。

第 1～5 epoch 只训练，不计入 wrong event。第 6～50 epoch 每轮训练结束后，额外进行一次确定性训练集推理：

- `model.eval()`、`torch.no_grad()`；
- 无随机增强、无 shuffle；每个稳定 `SampleID` 恰好出现一次；
- 记录硬预测 `pred[t, i]`，其中 `t=6,...,50`；
- 这条评估 loader 必须使用独立随机生成器，不能消耗训练增强和训练 sampler 的随机状态。这样额外评估不会改变后续训练轨迹。

对样本 `i` 定义：

```text
wrong[t, i] = 1(pred[t, i] != given_label[i])
wrong_rate[i] = sum_{t=6}^{50} wrong[t, i] / 45
```

另保存预测变化率，作为辅助稳定性证据：

```text
PC[t] = mean_i( pred[t, i] != pred[t-1, i] ),  t=7,...,50
```

Stage 1 的常规 `best_loss` 仍照现有定义由验证集无权重 loss 选择，用来和历史基线核对；它不用于选择 Stage 2 起点。

### 2.2 记录内容

每折导出一张逐样本表，至少包含：

```text
SampleID, given_label, clear_or_fuzzy, epoch_6_to_50_prediction,
wrong_count, wrong_rate, fold
```

同时输出每折的 `PC[t]` 曲线、正常/缺陷两类的 wrong-rate 直方图，以及正常缺陷分别的统计。不得把两类混在一个 BMM 中拟合。

## 3. BMM 诊断与放行门

### 3.1 拟合定义

每一折、每一标签类别分别对 `wrong_rate` 拟合：

- 单 Beta 模型：参数数 `k=2`；
- 两 Beta 混合模型：混合权重 1 个自由参数加两组 Beta 参数，`k=5`；
- 输入先截断到 `[1e-4, 1-1e-4]`，避免 Beta 在 0/1 处数值失效；
- 两成分拟合使用 5 个固定的确定性初值，EM 最多 200 次，收敛阈值 `1e-6`，取最大有限对数似然；
- 两成分按均值从小到大排序：低均值成分为 clean，高均值成分为 noise；
- BIC 使用 `BIC = -2 * log_likelihood + k * log(n)`。

对每个样本输出：

```text
tau_clean[i] = P(clean component | wrong_rate[i])
tau_noise[i] = P(noise component | wrong_rate[i])
```

这里的 `tau_noise` 仅是“训练轨迹异常概率”，不等于已证明的错标概率。

### 3.2 单折、单类别的结构门

下列条件必须同时满足，才认定该折该类别存在可用的双成分信号：

1. 两 Beta 拟合有限、收敛，且无 NaN/Inf/空成分；
2. `BIC_2 < BIC_1 - 10`；
3. 两个成分的有效样本量都不小于 `max(0.05 * n_class, 30)`；
4. 两成分均值差不小于 `0.10`；
5. 高均值成分确实对应更高的 wrong-rate，不能出现排序或 posterior 反转。

### 3.3 稳定性门

wrong rate 只有 45 个时间点且 epoch 间相关，单次 BIC 不足以证明存在真实双峰。因此再做两项固定检查：

1. **样本 Bootstrap：** 每个类别固定随机种子重采样 100 次。至少 90 次得到有效双 Beta；每次在原始样本上计算的 `tau_noise` 与全样本拟合结果的 Pearson 相关系数中位数必须不低于 `0.90`。
2. **时间半窗：** 分别用 epoch 6～27 和 epoch 28～50 计算 wrong rate，并各自拟合双 Beta。两半窗都必须通过 3.2 的结构门；在原始样本上得到的 noise posterior 与全窗口 posterior 的相关系数均不得低于 `0.70`。

任何一次拟合出现常数输入、分量塌缩、无法收敛或 posterior 无法定义，均视为该门失败；不以临时默认权重替代。

### 3.4 模糊缺陷保护门

这一步不参与训练，只防止再次把“真难缺陷”误判为噪声。

每折在缺陷类内部比较 clear-defect 与 fuzzy-defect 的 `tau_noise`：

```text
median(tau_noise_fuzzy_defect)
    <= median(tau_noise_clear_defect) + 0.05

P(tau_noise_fuzzy_defect >= 0.5)
    <= P(tau_noise_clear_defect >= 0.5) + 0.10
```

两项都满足才算该折通过。五折中至少 4 折通过；若有 2 折或更多折显示模糊缺陷系统性更像 noise，则停止 IDO。这个门直接吸取 RACW 失败的教训：困难程度不能被当成错标证据。

### 3.5 进入 Stage 2 的总条件

仅当以下条件全部成立，才允许跑 Stage 2：

- 五折的正常类和缺陷类均通过 3.2 和 3.3；
- 五折中至少 4 折通过 3.4；
- Stage 1 没有数值异常，且第 45～50 epoch 的平均 `PC[t]` 不高于第 6～10 epoch 的平均 `PC[t]`。

否则结论就是：**本数据上没有足够稳定的 IDO 前提，停止该路线。** 这仍是有价值的实验结论，不继续做参数补救。

## 4. Stage 2：唯一允许的 IDO-GS 训练

### 4.1 公平起点与优化规则

每一折从自己的 Stage 1 epoch-50 checkpoint 起步，复制出两条分支：

| 分支 | 训练 20 epoch | 目的 |
| --- | --- | --- |
| `CE-continue` | 原始 CE | 检验“多训练 20 轮”本身的收益 |
| `IDO-GS` | 下述 IDO 损失 | 检验 IDO 的额外收益 |

两分支使用相同初始权重、数据顺序、常规轻度增强、batch、AMP 设置和验证规则。两者都重新初始化同类型优化器，保留基线的 weight decay 和 betas；初始学习率固定为基线初始学习率的 `0.1`，并用新的 20 epoch cosine schedule 衰减到与基线相同的 `eta_min`。这个规则在开跑前写入配置，不随折调整。

每个分支都按既有的无权重验证 loss 保存自己的 best checkpoint。不得按模糊组、F1、外部测试结果挑选 checkpoint。

### 4.2 双视图和动态更新

同一个样本生成两个相互独立的现有轻度增强视图；不增加几何强增强，不允许破坏微小缺陷纹理。

Stage 2 第 `s` 轮训练前使用上一次确定性评估得到的 BMM posterior。训练结束后再做一次无增强训练集评估，将该轮 wrong event 追加到历史：

```text
n_observed = 45 + s
wrong_rate_s[i] = cumulative_wrong_count[i] / n_observed
```

再按第 3 节的规则重拟合 class-wise BMM，供下一轮使用。若任一折任一类别的 BMM 在 Stage 2 中失效，该折 IDO-GS 标记为无效并终止；不冻结旧 posterior，不偷偷退化成 CE。

### 4.3 损失定义

令两视图概率为 `q1`、`q2`，平均概率 `p_bar=(q1+q2)/2`，给定标签为 `y`。每个样本计算：

```text
L_C = tau_clean * 0.5 * (CE(q1, y) + CE(q2, y))
L_H = epsilon * ||q1 - q2||_2^2
L_N = tau_noise * confidence(p_bar) * entropy(p_bar)

epsilon = tau_clean * CDF_clean(wrong_rate)
        + tau_noise * (1 - CDF_noise(wrong_rate))

L_IDO = mean(L_C + L_H + L_N)
```

其中 `confidence(p_bar)=max(p_bar)`，`entropy(p_bar)=-sum(p_bar * log(p_bar+1e-8))`。`L_N` 使用置信度加权熵项，不使用“detach 后的伪标签交叉熵”这种梯度定义含糊的替代物。实现时需要锁定 IDO 官方仓库的 commit，并把该 commit、BMM 实现和所有数值保护写入脚本头部。

所有 posterior、CDF、损失权重均在当前 loss dtype 下显式转换；不能再出现 AMP 下 Float/Half 的 index assignment 类型不匹配。

## 5. 预注册判定标准

先只看五折结果和 OOF，不看锁定 3 板测试集。IDO-GS 必须同时满足：

1. 五折平均 F1 比 `CE-continue` 高至少 `0.30` 个百分点，且至少 3/5 折 F1 不低于 CE-continue；
2. 五折平均 AUC、AP 均不得比 CE-continue 低超过 `0.10` 个百分点；
3. OOF 模糊缺陷召回不低于 CE-continue，且不低于已有固定权重 1.0 的 `54.86%`；
4. 不得有任一折的模糊缺陷召回比 CE-continue 低 `10` 个百分点或更多；
5. 无 BMM 失效、NaN、类别塌缩或其它训练中断。

所有条件满足，才用五折各自最佳 checkpoint 对锁定 3 板执行 **一次** 外部测试；不得根据外部测试回调任何参数。任一条件失败，结论为“IDO-GS 不优于当前路线”，停止，不开展第二轮调参。

## 6. 必须输出的结果

每折分别列出，不只给均值：

- Acc、Precision、Recall、F1、Specificity、AUC、AP；
- clear-normal、clear-defect、fuzzy-normal、fuzzy-defect 四组的样本数、正确数、召回/正确率；
- Stage 1 的 BIC、成分均值、有效样本量、Bootstrap/半窗稳定性、模糊缺陷保护门；
- Stage 2 的每 epoch BMM 状态和最佳 epoch；
- `IDO-GS`、`CE-continue`、固定权重 1.0 三者的五折均值对照。

锁定测试仅在通过第 5 节时输出，且仍把正常与缺陷分开报告。

## 7. 工作量与止损

- 必做：五折 Stage 1，50 epoch，外加确定性训练集评估记录。
- 条件必做：仅当全部门通过时，五折的 IDO-GS 20 epoch + 五折 CE-continue 20 epoch。
- 禁止：BMM 阈值扫描、权重扫描、增强扫描、因单折差而追加实验。

这是一次有明确止损点的验证：它要么证明“训练轨迹中的噪声分离”在 Head40 上可用，要么干净地排除该假设，而不是继续消耗实验次数。
