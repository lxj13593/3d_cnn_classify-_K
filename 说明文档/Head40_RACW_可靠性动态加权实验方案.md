# Head40-533 RACW 可靠性动态加权实验方案

## 1. 实验目的

在不改变 Head40-533 网络结构、数据划分和评价协议的前提下，对 `fuzzy` 样本进行样本级动态加权。

任务始终为二分类：

```text
normal = 0
defective = 1
```

`fuzzy` 不是第三类，也不在第一阶段修改其标签。

## 2. 已有结论

```text
ambiguous weight = 1.0：整体最稳定
ambiguous weight = 0.5：三板 Recall 有提升，但五折验证未稳定超过 1.0
ambiguous weight = 0.0：验证和三板测试均明显下降
```

因此，模糊样本不能整体删除；但统一固定权重也过于粗糙。RACW 只估计每个模糊样本对原标签的可靠程度，再决定其损失权重。

## 3. 方案名称

```text
RACW = Reliability-Aware Centroid Weighting
```

第一阶段仅包含：

```text
高可靠清晰样本类中心
→ 模糊样本的特征空间可靠度
→ 动态损失权重
→ hard-label CrossEntropyLoss
```

不包含：

```text
Teacher
soft label
pseudo label
label correction
额外网络模块
```

## 4. 保持不变的部分

```text
Head40-533
相同五折划分与随机种子
相同数据增强、优化器、学习率、batch size、50 epoch
相同 best_loss 检查点规则
相同锁定三板测试集
```

外部三板只用于最终评价，绝不根据其结果调整参数。

## 5. 样本定义

```text
动态加权对象：difficulty == "fuzzy"
```

第一版用于建立类中心的候选样本严格限定为：

```text
difficulty == "clear"
```

`reviewed_defect` 与 `supplemental` 保持原有 hard label 和训练权重 1.0，但第一版不参与类中心更新。这样“清晰类中心”的来源明确，不把不同来源混在一起。

## 6. Warm-up 与类中心初始化

### Epoch 1–5

全部训练样本使用普通 CE：

```text
weight = 1.0
```

### Warm-up 后初始化

在第 5 个 epoch 结束后，仅对当前训练折做一次：

```text
model.eval()
torch.no_grad()
无数据增强
```

对 `difficulty == "clear"` 的样本，只有同时满足下列条件才进入对应类别中心：

```text
预测类别 == 原始标签
原始标签的预测概率 >= p_thr
```

第一版固定：

```text
p_thr = 0.8
```

对 normal 与 defective 分别平均其 L2 归一化后的 Layer4 GAP 512 维特征，得到初始中心：

```text
C0 = normalize(mean(z_clear_normal))
C1 = normalize(mean(z_clear_defective))
```

每一类至少需要 10 个合格样本；若任一类不足 10 个，则本折报错并停止；不得使用零中心或伪造中心。

## 7. 类中心更新

每个 epoch 开始前，在该训练折、无增强、`eval + no_grad` 模式下重新筛选高可靠 `clear` 样本，并以 EMA 更新中心：

```text
Ck = normalize(mu * Ck + (1 - mu) * mean(z_high_reliability_clear_k))
```

固定：

```text
mu = 0.99
```

若某类别本 epoch 没有合格样本，则保留上一轮中心，不更新。

中心、特征和可靠度计算都必须脱离反向传播图。

## 8. 模糊样本可靠度

对训练批次中的 `fuzzy` 样本，提取并 L2 归一化 Layer4 GAP 特征 `z_i`。

```text
s0 = cosine(z_i, C0)
s1 = cosine(z_i, C1)
```

若原标签为 `y`，另一类为 `1-y`：

```text
margin_i = s_y - s_(1-y)
r_i = sigmoid(margin_i / tau)
```

固定：

```text
tau = 0.2
```

`r_i` 越高，表示特征空间越支持原标签；越低，表示样本位于边界附近或更接近另一类中心。

## 9. 动态权重与损失

```text
clear / reviewed_defect / supplemental：w_i = 1.0
fuzzy：w_i = w_min + (1 - w_min) * r_i
```

第一版固定：

```text
w_min = 0.3
```

使用逐样本交叉熵：

```python
criterion = CrossEntropyLoss(reduction="none")
loss_i = criterion(logits_i, label_i)
loss = sum(w_i * loss_i) / sum(w_i)
```

权重 `w_i` 必须 `detach`；其作用是调节损失，不反向优化类中心或可靠度本身。

## 10. 训练顺序

```text
Epoch 1–5：普通 CE，所有样本权重 1.0
Epoch 5 后：初始化 C0、C1
Epoch 6–50：每 epoch 更新冻结类中心，再对 fuzzy 使用 RACW 动态权重
```

训练时用当前批次的特征计算权重；权重计算不参与梯度。类中心仅由本折训练数据更新，验证集和外部测试集绝不参与。

## 11. 必须记录的日志

每折、每个 epoch 记录：

```text
fuzzy 总数
fuzzy-normal / fuzzy-defect 的平均、最小、最大、标准差权重
高可靠 clear-normal / clear-defect 样本数
C0 与 C1 的余弦相似度
两个中心是否完成初始化
```

验证时额外按以下四组分别保存预测、正确数和错误数：

```text
clear-normal
clear-defect
fuzzy-normal
fuzzy-defect
```

其中正常组报告正确率与 FP；缺陷组报告 Recall 与 FN。不要在单一真实类别子集上解释 F1。

## 12. 评价与判定

主检查点保持：

```text
best_loss（未加权验证 Loss）
```

五折验证报告：

```text
Loss
Accuracy
Precision
Recall
F1
Specificity
AUC
AP
```

预先定义第一阶段的正向信号：

```text
五折验证 F1 不低于 1.0 CE 基线
且 AUC、AP 不下降
且 fuzzy-defect Recall 不低于 1.0 CE 基线
```

只有出现正向信号，才运行锁定三板测试；随后才考虑 Teacher 或 soft label 的第二阶段。

若 RACW 明显低于 1.0 CE，则停止该方向，不对 `p_thr`、`tau`、`w_min`、`mu` 做大范围搜索。

## 13. 实验命名

本实验的代码与结果统一使用 RACW 名称，不占用、不覆盖已有的 E1～E4 文件。

~~~text
对照：Head40-533 + fixed fuzzy weight = 1.0 / 0.5 / 0.0
本实验：Head40-533 + RACW
~~~

## 14. 第二阶段条件

仅当 E3 的五折验证满足第 12 节的正向信号，才进一步研究：

```text
EMA Teacher
动态 soft label
Teacher Pool
robust loss 对照
```

第一阶段未通过时，不进入标签修正阶段。
