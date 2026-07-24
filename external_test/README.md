# 三块板外部测试流程

本文档记录当前项目对外部三块板 `1022`、`1077`、`1305` 的完整测试流程。测试对象为：

1. 原始全长度基线 3D ResNet18。
2. 头部区域 head37 3D ResNet18。

两个模型分开测试。每个模型、每一折都会分别测试“验证集 F1 最佳”“验证集 Loss 最佳”和“最后一轮”三种检查点。head37 数据必须提前裁好，正式测试脚本不会进行二维定位或在线裁剪。

## 1. 重要约定

- 正常类别：`normal`，标签为 `0`。
- 缺陷类别：`defective`，标签为 `1`。
- 默认分类阈值：`0.5`。
- 默认检查点类型：`best_f1`、`best_loss`、`last`，三类结果独立统计。
- 外部板：`1022`、`1077`、`1305`。
- 每块板固定测试 `490` 个已标注钻孔，总计 `1470` 个。
- RAW 数据类型：`uint8`。
- 文件名中的尺寸顺序是 `W_H_D`，读入 NumPy 后的形状顺序是 `D x H x W`。
- 例如文件名中的 `_29_29_237-`，实际送入网络的数组形状为 `(237, 29, 29)`。
- 头部定位规则默认钻孔头部位于二维图右侧。方向不统一时，必须先统一方向再重新生成头部数据。
- 外部测试集只用于最终泛化性评估，不用于重新选择模型、调阈值或修改裁剪参数。

## 2. 相关文件

```text
external_test/
├── README.md
├── test_external_3boards_resnet18_all_checkpoints.py
└── test_external_3boards_resnet18_head37_all_checkpoints.py

data_operate/
├── prepare_external_head37_by_2d_picture.py
└── data_load_external_variable_shape.py

model/
├── resnet_18_34_50_3d.py
└── resnet_18_34_50_3d_head_region_37.py
```

用途说明：

- `prepare_external_head37_by_2d_picture.py`：根据配套二维图定位右侧头部，一次性生成预裁剪 RAW。
- `data_load_external_variable_shape.py`：只读取 RAW、检查数据并归一化，不进行裁剪。
- `test_external_3boards_resnet18_all_checkpoints.py`：只测试原始全长度 ResNet18 的三类检查点。
- `test_external_3boards_resnet18_head37_all_checkpoints.py`：只测试提前裁好的 head37 数据及对应三类检查点。

## 3. Python 环境

项目当前使用的 PyTorch 环境为：

```text
D:\anconda\envs\pytorch-2.7.1-gpu\python.exe
```

先进入项目根目录：

```powershell
Set-Location 'E:\pythonproject\3d_cnn_classify _K'
```

不要直接使用系统默认 `python`，因为默认环境可能没有安装 PyTorch。

## 4. 原始外部数据结构

原始外部数据放在：

```text
datasets/external_test/
├── 1022/
├── 1077/
└── 1305/
```

每块板至少需要以下内容：

```text
<board>/
├── drill_normal_3d/       # 已标注正常 RAW
├── drill_defect_3d/       # 已标注缺陷 RAW
├── drill_feature/         # 与 RAW 配对的二维 JPG
└── manifest.csv           # RAW、二维图、样本顺序的对应关系
```

原始 ResNet18 测试只读取 `drill_normal_3d`、`drill_defect_3d` 和 `manifest.csv`。头部预裁剪还会读取 `drill_feature` 中的二维图。板目录中的其他副本不会作为正式测试输入。

当前数据统计如下：

| 板号 | 总数 | 正常 | 缺陷 | 原始体数据形状 | head37 输入形状 |
|---|---:|---:|---:|---|---|
| 1022 | 490 | 401 | 89 | 239 个 `237x29x29`，251 个 `296x37x37` | 239 个 `29x29x29`，251 个 `37x37x37` |
| 1077 | 490 | 389 | 101 | 490 个 `296x37x37` | 490 个 `37x37x37` |
| 1305 | 490 | 457 | 33 | 490 个 `296x37x37` | 490 个 `37x37x37` |
| 合计 | 1470 | 1247 | 223 | 239 个小尺寸，1231 个标准尺寸 | 239 个 `29^3`，1231 个 `37^3` |

1022 中的外部钻孔横截面较小，因此保留原始 `29x29` 横截面，不插值到 `37x37`。两个网络都带有自适应全局平均池化，可以在结构上接收这些可变尺寸；尺寸变化本身属于外部泛化测试的一部分，结果文件会额外按原始形状分组统计。

## 5. 提前生成 head37 数据

### 5.1 什么时候需要运行

以下情况需要重新运行预裁剪：

- 第一次准备外部 head37 测试数据。
- 更换了原始 RAW、二维图或标签。
- 调整了钻孔方向。
- 修改了二维前景阈值或头部定位规则。

原始数据没有变化时，不需要在每次测试前重复裁剪。

### 5.2 运行命令

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' data_operate\prepare_external_head37_by_2d_picture.py
```

默认输入：

```text
datasets/external_test
```

默认输出：

```text
datasets/external_test_head37
```

也可以指定其他目录或板号：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' data_operate\prepare_external_head37_by_2d_picture.py `
  --source-root datasets\external_test `
  --output-root datasets\external_test_head37 `
  --boards 1022 1077 1305
```

### 5.3 头部定位与裁剪规则

每个样本使用 `manifest.csv` 找到 RAW 对应的二维 JPG，然后按以下统一规则处理：

1. 将二维图转为灰度图。
2. 灰度值大于 `20` 的像素视为前景。
3. 某列只要存在一个前景像素，该列就视为前景列。
4. 最右侧前景列记为 `HeadEnd`，也就是当前规则中的头部末端。
5. 裁剪长度等于横截面边长：横截面为 `29x29` 时截取 29 层，横截面为 `37x37` 时截取 37 层。
6. 设裁剪长度为 `L`，则裁剪范围为 `[HeadEnd + 1 - L, HeadEnd + 1)`。
7. 如果左侧不足 `L` 层，在裁剪结果左侧补零；当前清单会把补零数量记录为 `LeftPad`。
8. 不做插值、不改变灰度值，也不从测试脚本中再次裁剪。

输出形状：

- `(237, 29, 29)` 变为 `(29, 29, 29)`。
- `(296, 37, 37)` 变为 `(37, 37, 37)`。

### 5.4 裁剪检查图

每块板的检查图位于：

```text
datasets/external_test_head37/<board>/check_images/
```

线条含义：

- 绿色竖线：裁剪起点 `CropStart`，该列包含在裁剪区域中。
- 蓝色竖线：最右侧前景列 `HeadEnd`，该列包含在裁剪区域中。
- 绿色到蓝色之间的淡绿色区域：实际截取范围。
- 实际数组使用右开区间，结束位置为 `CropEndExclusive = HeadEnd + 1`，因此蓝线所在列不会丢失。
- 当前检查图没有红色定位线。

建议每块板至少检查以下样本：

- 正常和缺陷各若干个。
- 位于二维图不同位置的样本。
- 1022 中的 `29x29` 与 `37x37` 两种尺寸。
- 接近图像左右边界的样本。

如果头部仍然出现在裁剪区域右侧之外，先检查原图方向是否统一，而不是直接运行测试。

### 5.5 预裁剪输出结构

```text
datasets/external_test_head37/<board>/
├── normal/
├── defective/
├── check_images/
└── manifest_head.csv
```

`manifest_head.csv` 记录源文件、输出文件、标签、源形状、输出形状、二维图文件、前景范围和裁剪位置。它是之后核对两个测试集是否对应同一批样本的主要依据。

预处理脚本不会修改或删除 `datasets/external_test` 中的原始数据。

## 6. 准备五折模型权重

两个测试脚本默认读取每折的三类权重。每个模型需要 `5 折 x 3 类 = 15` 个检查点，两个模型合计 30 个。默认目录必须是：

```text
model_best_last/
├── resnet18_5fold/
│   ├── fold_0/
│   │   ├── best_resnet18_3d.pth
│   │   ├── best_loss_resnet18_3d.pth
│   │   └── last_resnet18_3d.pth
│   ├── fold_1/                 # 同样三个文件
│   ├── fold_2/                 # 同样三个文件
│   ├── fold_3/                 # 同样三个文件
│   └── fold_4/                 # 同样三个文件
└── resnet18_head_region_37_5fold_ce/
    ├── fold_0/
    │   ├── best_resnet18_head_region_37_3d.pth
    │   ├── best_loss_resnet18_head_region_37_3d.pth
    │   └── last_resnet18_head_region_37_3d.pth
    ├── fold_1/                 # 同样三个文件
    ├── fold_2/                 # 同样三个文件
    ├── fold_3/                 # 同样三个文件
    └── fold_4/                 # 同样三个文件
```

三类检查点含义：

- `best_f1`：该折验证集 F1 最高时保存的权重。
- `best_loss`：该折验证集 Loss 最低时保存的权重。
- `last`：该折训练最后一轮保存的权重。

每个检查点必须包含：

- `model_state_dict`
- `norm_params.global_min`
- `norm_params.global_max`

测试脚本不允许用外部测试集重新计算归一化参数，也没有外部统计量的备用逻辑。检查点缺少训练归一化参数时会直接报错。

## 7. 测试原始全长度 ResNet18

运行：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_all_checkpoints.py
```

默认输入：

```text
datasets/external_test
```

默认权重：

```text
model_best_last/resnet18_5fold/fold_<0-4>/
├── best_resnet18_3d.pth
├── best_loss_resnet18_3d.pth
└── last_resnet18_3d.pth
```

默认结果目录：

```text
test_result/external_3boards_resnet18_all_checkpoints
```

这个脚本将原始完整 RAW 直接送入基线 ResNet18，不截取头部，也不重采样。模型文件为 `model/resnet_18_34_50_3d.py`。

## 8. 测试 head37 ResNet18

确认 head37 数据已经生成并检查后运行：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_head37_all_checkpoints.py
```

默认输入：

```text
datasets/external_test_head37
```

默认权重：

```text
model_best_last/resnet18_head_region_37_5fold_ce/fold_<0-4>/
├── best_resnet18_head_region_37_3d.pth
├── best_loss_resnet18_head_region_37_3d.pth
└── last_resnet18_head_region_37_3d.pth
```

默认结果目录：

```text
test_result/external_3boards_resnet18_head37_all_checkpoints
```

模型文件为 `model/resnet_18_34_50_3d_head_region_37.py`。该版本在 stem 阶段的长度、高度和宽度方向都使用步长 1，避免短头部输入在网络最前面压缩过快；之后 `layer2`、`layer3`、`layer4` 再进行下采样。

正式测试脚本只读取 `normal/`、`defective/` 和 `manifest_head.csv` 中已经裁好的 RAW，不读取二维图，也不包含头部定位代码。

## 9. 测试阶段实际执行的步骤

两个测试脚本的测试逻辑一致，只是模型、输入数据和权重路径不同：

1. 检查三块板目录是否存在。
2. 检查选择的五折索引是否有效。
3. 检查每一折的 `best_f1`、`best_loss` 和 `last` 权重是否全部存在。
4. 根据 manifest 检查每块板是否正好有 490 个唯一标注样本，并拒绝重复源文件或重复 `SampleOrder`。
5. 检查 RAW 文件名尺寸、manifest 尺寸和实际字节数是否一致。
6. 依次加载当前折的三类权重及各检查点保存的训练集归一化参数；参数必须是有限数，并且 `global_max > global_min`。
7. 使用 `(x - global_min) / (global_max - global_min + 1e-6)` 归一化，并截断到 `[0, 1]`。
8. 使用 `batch_size=1` 读取数据，以支持 29 和 37 横截面的可变形状。
9. 模型进入 `eval()`，关闭梯度；CUDA 可用时启用自动混合精度推理。
10. 对每个样本计算 softmax 的缺陷类别概率。
11. 每一种检查点、每一折、每一块板分别计算指标。
12. 在同一种检查点内部，对同一样本的五折缺陷概率求算术平均，再使用阈值 `0.5` 得到该检查点的最终集成预测。
13. `best_f1`、`best_loss`、`last` 之间不进行概率平均，也不合并成一个预测。
14. 分别输出三类检查点在每块板、三块板合计和不同原始尺寸子集上的结果。

五折随机种子为：

```text
[42, 123, 2026, 3407, 777]
```

测试时设置随机种子主要用于保持运行环境可重复；因为模型处于 `eval()` 状态，Dropout 不参与随机丢弃。

## 10. 可选命令参数

查看帮助：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_all_checkpoints.py --help
```

只测试一块板和一折，用于快速检查流程：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_all_checkpoints.py `
  --boards 1022 `
  --folds 0 `
  --checkpoint-kinds best_f1 best_loss last `
  --device cuda
```

head37 的快速检查：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_head37_all_checkpoints.py `
  --boards 1022 `
  --folds 0 `
  --checkpoint-kinds best_f1 best_loss last `
  --device cuda
```

主要参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--data-root` | 对应模型的数据根目录 | 脚本内置目录 |
| `--model-root` | 五折模型根目录 | `model_best_last` |
| `--output-dir` | 结果输出目录 | 各模型独立目录 |
| `--boards` | 要测试的板号 | `1022 1077 1305` |
| `--folds` | 要测试的折号 | `0 1 2 3 4` |
| `--checkpoint-kinds` | 独立测试的检查点类型 | `best_f1 best_loss last` |
| `--threshold` | 缺陷概率阈值 | `0.5` |
| `--num-workers` | DataLoader 进程数 | `0` |
| `--device` | `auto`、`cpu` 或 `cuda` | `auto` |

快速检查只用于确认程序能运行，正式结果必须重新使用三块板和五折完整命令。不要根据外部测试表现反复调整 `--threshold`。

## 11. 输出文件及其用途

每个模型的结果目录都会生成：

| 文件 | 内容 |
|---|---|
| `per_fold_predictions.csv` | 每种检查点、每一折对每个样本的缺陷概率和预测结果 |
| `per_fold_metrics.csv` | 每种检查点、每一折、三块板各自及三块板合并后的分类指标 |
| `fivefold_mean_std.csv` | 按 `Board + Metric` 分块，三种权重分别为列，数值采用 `88.24% ± 1.14%` |
| `fivefold_mean_std_raw.csv` | 三块板各自及合并结果的原始小数值，均值与标准差分列保存 |
| `fivefold_summary_by_board/board_<板号>.csv` | 每块板单独一张“指标为行、三种权重为列”的五折汇总表 |
| `fivefold_summary_by_board/all_boards.csv` | 1470个样本合并后重新计算的五折总表 |
| `ensemble_predictions.csv` | 每种检查点内部对每个样本进行五折概率平均后的预测 |
| `ensemble_metrics_board_and_shape.csv` | 三类检查点各自在每块板、全部板和形状子集上的五折集成指标 |
| `external_test_summary.xlsx` | 包含 `Board 1022`、`Board 1077`、`Board 1305`、`All boards` 四张同版式汇总表及原始明细 |
| `<检查点类型>/board_<板号>_confusion.png` | 对应检查点在每块板上的五折集成混淆矩阵 |
| `<检查点类型>/all_boards_confusion.png` | 对应检查点在三块板合并后的五折集成混淆矩阵 |

逐折、原始数值和集成明细使用 `CheckpointKind` 列区分权重；展示版五折汇总则直接以 `best_f1`、`best_loss`、`last` 作为三列。

建议重点查看：

1. `ensemble_metrics_board_and_shape.csv` 中按 `CheckpointKind` 区分的三类结果。
2. `Subset=Board` 的三块板结果与 `Subset=All boards` 的总体外部泛化结果。
3. `fivefold_summary_by_board` 中每块板和总表的 `均值 ± 标准差`；标准差越小，五折越稳定。
4. `ensemble_predictions.csv` 中各检查点的具体误判样本及其概率。
5. 1022 的不同 `RawShape` 子集，判断小尺寸钻孔是否造成明显性能变化。

指标解释：

- `Accuracy`：全部样本预测正确的比例。
- `Precision`：预测为缺陷的样本中，真正缺陷的比例。
- `Recall`：真实缺陷中被检出的比例，也就是缺陷敏感度。
- `Specificity`：真实正常中被正确识别的比例。
- `F1`：Precision 与 Recall 的调和平均。
- `AUC`：不同阈值下区分正常和缺陷的能力。
- `AP`：以缺陷为正类的平均精确率，适合当前缺陷样本较少的情况。

## 12. 数据泄露检查

当前流程在以下前提下不存在训练数据泄露：

- 三块外部板没有参与任何模型训练、验证或五折划分。
- 模型和阈值在查看外部测试结果之前已经确定。
- 归一化参数只来自各折训练检查点，不使用外部测试集统计量。
- 头部边界由二维图的几何前景统一确定，不根据正常或缺陷标签调整位置。
- 标签只用于组织测试数据和计算最终指标，不参与模型输入或概率计算。
- 原始模型与 head37 模型对应同一批外部样本，head37 只是这些样本的确定性预裁剪版本。

需要特别注意：如果根据这三块板的结果继续选择模型、调阈值、改裁剪长度或决定数据增强方案，那么这三块板就被实际用作验证集，不能再把同一结果当作完全独立的最终测试结果。此时应另准备新的外部测试板。

## 13. 运行前检查清单

- [ ] 当前目录是项目根目录。
- [ ] 使用 `pytorch-2.7.1-gpu` 环境，而不是系统默认 Python。
- [ ] 三块板方向一致，头部均在二维图右侧。
- [ ] 每块板有 490 个唯一标注样本。
- [ ] 正常和缺陷 RAW 放在正确的类别目录。
- [ ] `manifest.csv` 中 RAW 与二维 JPG 一一对应。
- [ ] head37 数据已提前生成。
- [ ] 已抽查 `check_images` 中的绿色起点和蓝色末端。
- [ ] 两个模型五折中的 `best_f1`、`best_loss`、`last` 共 30 个权重均已恢复。
- [ ] 每个检查点包含训练归一化参数。
- [ ] 正式测试使用全部三块板和全部五折。
- [ ] 测试前已确定阈值，不根据结果临时调参。
- [ ] 需要保留旧结果时，先备份或更换 `--output-dir`。

## 14. 常见报错

### `ModuleNotFoundError: No module named 'torch'`

使用了错误的 Python。改用：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' <脚本路径>
```

### `Missing checkpoint`

对应折的一种或多种权重不在默认目录。检查第 6 节所列的完整目录和文件名，确认 `best_f1`、`best_loss`、`last` 三类文件都存在。

### `Checkpoint has no training normalization parameters`

权重文件没有保存有效的 `norm_params`，或者出现非有限数、`global_max <= global_min`。当前测试流程会拒绝使用外部数据统计量代替，需要恢复训练脚本正常保存的完整检查点。

### `has ... samples; expected 490`

该板的类别目录、manifest 或预裁剪输出不完整，或者混入了多余 RAW。先核对文件数量和 manifest，不要直接修改代码中的 490 绕过检查。

### `RAW byte count mismatch`

文件名尺寸、manifest 尺寸或实际 RAW 内容不一致。检查文件是否复制错误、截断或命名错误。

### `Picture/raw depth mismatch`

预裁剪时，二维图宽度与 RAW 长度方向不一致。检查 `manifest.csv` 是否把错误的二维图与 RAW 配对。

### 重新预裁剪时报 `stale` 输出

输出目录中存在不属于当前输入清单的旧文件。确认原始数据和标签无误后，备份并清理对应板的旧 head37 输出，再重新生成。不要删除 `datasets/external_test` 中的原始数据。

### Excel 没有生成

CSV 仍然是完整结果。Excel 输出依赖 pandas 可用的 Excel 写入引擎；安装对应引擎后可重新运行，或直接使用 CSV。

## 15. Git 与备份

项目 `.gitignore` 已屏蔽以下内容：

- `datasets/`
- `model_best_last/`
- `test_result/`
- `train_val_result/`
- `*.pth`、`*.pt`、`*.ckpt`
- Python 缓存

因此 Git 提交只会保存代码和本文档，不会保存外部数据、预裁剪数据、模型权重和测试结果。执行 Git 恢复、换电脑或清理项目之前，必须单独备份这些目录。

## 16. 最简正式执行顺序

仅在外部原始数据发生变化时重新生成 head37：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' data_operate\prepare_external_head37_by_2d_picture.py
```

测试原始全长度 ResNet18：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_all_checkpoints.py
```

测试 head37 ResNet18：

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' external_test\test_external_3boards_resnet18_head37_all_checkpoints.py
```

最后分别查看：

```text
test_result/external_3boards_resnet18_all_checkpoints
test_result/external_3boards_resnet18_head37_all_checkpoints
```
