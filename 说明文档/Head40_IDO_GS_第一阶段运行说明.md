# Head40 IDO-GS 第一阶段运行说明

本次按“全部独立新写”实现。四个新 Python 文件彼此配合，不导入旧实验的加载器、模型或训练代码。原始数据和已经准备好的五折划分继续使用。

## 需要拷到新电脑的文件

保持下面的相对目录，放入新电脑的项目根目录：

~~~text
项目根目录/
  data_operate/
    data_load_head40_ido_gs.py
  model/
    resnet18_3d_head40_stem533_ido_gs.py
  train_val/
    main_head40_ido_gs_stage1_5fold.py
    ido_gs_stage1_diagnostics.py
  datasets/
    resnet18_head40_toml_direction_unified_multiboard_5fold/
      split_config.json
      fold_0/train.csv、val.csv
      ...
      fold_4/train.csv、val.csv
~~~

前四个是新代码；datasets 内使用此前已经准备好的划分，不重新生成。运行环境需要 PyTorch、NumPy、SciPy、scikit-learn、tqdm；训练使用新电脑原有的 PyTorch 环境。

## 直接运行

在 IDE 中直接运行 train_val/main_head40_ido_gs_stage1_5fold.py，默认依次执行五折。

也可以在项目根目录执行：

~~~powershell
python train_val/main_head40_ido_gs_stage1_5fold.py
~~~

默认每折：50 epoch、batch size 4、AdamW、lr=1e-4、weight decay=1e-3、cosine、eta_min=1e-7。第一阶段全部样本权重均为 1，不进行 IDO 动态降权。

数据路径查找顺序保持原规则：命令行指定路径 → DRILL_HEAD40_TOML_TRAIN_ROOT 环境变量 → 项目 datasets 目录 → 原 split_config 保存的路径 → 各盘符下原有固定数据目录。

因此新电脑原数据位置没有变时，通常无需设置参数。若无法自动定位，可明确指定：

~~~powershell
python train_val/main_head40_ido_gs_stage1_5fold.py --dataset-root "D:\实际数据位置\multiboard_head40_toml"
~~~

上面的 D 盘路径是示例，要替换为实际位置。若划分放在不同位置，加 --split-root "实际五折划分目录"。

## 运行过程中会做什么

第 1～5 轮普通训练与验证；第 6～50 轮另外对本折训练集做一次无增强预测，累计每个样本的错误次数。新增评估隔离随机状态，不用于梯度更新，也不读取三板测试集。

每折完成后，CPU 执行 BMM 诊断。网络训练此时已经完成，Bootstrap 的进度会单独显示。结构门已经失败的类别会跳过 Bootstrap。

五折普通 CE 都会完成，然后给出总判定。本入口不会自动进入第二阶段。模型结构与 Head40-533 相同，默认参数量为 33,141,954。

## 中断后继续

~~~powershell
python train_val/main_head40_ido_gs_stage1_5fold.py --resume
~~~

每轮保存模型、优化器、学习率调度器、GradScaler、随机状态和完整错误轨迹，支持从上次完整保存的 epoch 继续。第 50 轮完成后，stage1_last.pth 转为 stage1_epoch50.pth。未完整写出的当前轮会重新执行。

默认 num_workers 固定为 0，保证训练增强随机状态可恢复。续训时保持 AMP、deterministic 等开关与原运行一致。已有输出目录不会被当作全新实验覆盖；有完整 checkpoint 时使用 --resume。

只运行指定折：

~~~powershell
python train_val/main_head40_ido_gs_stage1_5fold.py --folds 0
~~~

不足五折时，总判定为 incomplete，不会给出通过结论。完成所有折后才输出五折均值。

## 已训练完，只做诊断

~~~powershell
python train_val/main_head40_ido_gs_stage1_5fold.py --diagnose-only
~~~

该模式读取完整训练记录，不加载模型和 RAW 数据，不需要 PyTorch。需要 NumPy、SciPy、scikit-learn、tqdm。记录、诊断实现均未改变时，直接读取已有诊断，不重复拟合。

迁移到其他目录时，保留完整结果文件夹，并通过 --result-root 指定位置。BMM 参数保存在对应诊断 JSON 中，与轨迹和 fold 身份关联，不为它们重复改写大型模型文件。

## 去哪里看结果

默认权重目录：

~~~text
model_best_last/head40_stem533_ido_gs_stage1_22boards_5fold/
~~~

默认结果目录：

~~~text
train_val_result/head40_stem533_ido_gs_stage1_22boards_5fold/
~~~

主要看：

| 文件 | 用途 |
| --- | --- |
| stage1_gate.json | 五折整体 pass / fail / incomplete，以及每折原因 |
| comparison_mean.csv | 五折 best_loss 的主要指标均值 |
| fivefold_mean_std.csv / .xlsx | 与既有 Head40-533 实验相同的 `best_f1`、`best_loss`、`last` 五折均值±标准差表 |
| fivefold_checkpoint_metrics.csv / .xlsx | 与既有实验相同的逐折、逐 checkpoint 原始指标 |
| fold_k/checkpoint_metrics.csv、validation_metrics_*.csv、validation_predictions_*.csv | 与既有实验相同的逐折结果文件及字段；原有 Stage 1 专用文件仍保留 |
| oof_metrics_best_loss.csv / .json、oof_metrics_best_loss_by_board*.csv | 与既有实验相同的 OOF 总体及分板指标 |
| legacy_export_status.json | 记录历史结果中哪些 checkpoint 实际存在；缺失项留空，不伪造 |
| comparison_folds.csv | 各折 best_loss 指标 |
| comparison_groups.csv | 每折清晰/模糊、正常/缺陷的分开计数 |
| oof_groups_best_loss.csv | OOF 分组检出数；模糊缺陷总体应为 144 个 |
| fold_k/stage1_group_diagnostics.csv | 各组实际 CE 系数，重点看模糊缺陷 |
| fold_k/stage1_bmm_diagnostics.json | BMM、Bootstrap、半窗和具体失败原因 |
| fold_k/stage1_epoch_predictions.csv | 第 6～50 轮逐样本预测 |

pass 只表示符合本次第二阶段准入条件，不表示已经获得准确率提升。fail 表示本次诊断未通过；不会把失败折删掉后再报五折均值。

第一阶段结束后，先读取 stage1_gate.json 和 comparison_mean.csv，再按最终方案决定是否实现、运行第二阶段。

旧版 Stage 1 已完成运行只保存了 `best_loss` 预测和权重。使用修订后的脚本可从已有文件补齐 `best_loss` 的旧格式结果，但不能从训练历史恢复未保存的 `best_f1` 和 `last` 验证预测；要得到完整三套 checkpoint 结果，需在新代码下重新训练。

## 本地验证范围

本机未安装 PyTorch，因此已进行的验证以语法、独立依赖关系、CPU 数值诊断和轨迹/结果处理为主；尚未在本机验证真实 RAW 的张量加载、GPU 前向或完整训练。没有在本机启动实际实验。
