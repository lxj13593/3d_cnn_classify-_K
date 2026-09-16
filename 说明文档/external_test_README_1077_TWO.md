# 1077_two external-board test

This workflow is dedicated to external board `1077_two`. It does
not read boards `1022` or `1305`, and it does not modify the existing
three-board test scripts.

## 1. Prepare the Head37 data

Run from the project root:

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' `
  data_operate\prepare_external_1077_two_head37.py
```

Input:

```text
datasets/external_test/1077_two
```

Output:

```text
datasets/external_test_head37/1077_two
|-- normal/
|-- defective/
|-- check_images/
`-- manifest_head.csv
```

The locator uses the original unchanged rule: convert the paired 2D picture to
grayscale, apply threshold `20`, and treat a column as foreground when that
column contains at least one foreground pixel. `HeadEnd` is the rightmost such
column. The output RAW contains `[HeadEnd + 1 - 37, HeadEnd + 1)` and therefore
has shape `37 x 37 x 37`.

The class label only determines the output class directory. It is not used to
choose the crop coordinate.

## 2. Test the original full-volume ResNet18

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' `
  external_test\test_external_1077_two_resnet18_all_checkpoints.py
```

Default result directory:

```text
test_result/external_1077_two_resnet18_all_checkpoints
```

## 3. Test the Head37 ResNet18

```powershell
& 'D:\anconda\envs\pytorch-2.7.1-gpu\python.exe' `
  external_test\test_external_1077_two_resnet18_head37_all_checkpoints.py
```

Default result directory:

```text
test_result/external_1077_two_resnet18_head37_all_checkpoints
```

## 4. Checkpoint and normalization policy

Both test scripts independently evaluate all five folds and these three
checkpoint types:

```text
best_f1
best_loss
last
```

Each checkpoint must contain `model_state_dict` and `norm_params`. Test data is
normalized only with that checkpoint's training `global_min` and `global_max`.
The scripts never calculate replacement normalization statistics from board
`1077_two`.

## 5. Saved results

Each model has its own result directory containing:

```text
per_fold_predictions.csv
per_fold_metrics.csv
fivefold_mean_std.csv
fivefold_mean_std_raw.csv
ensemble_predictions.csv
ensemble_metrics.csv
external_test_summary.xlsx
fivefold_summary_by_board/board_1077_two.csv
best_f1/board_1077_two_confusion.png
best_loss/board_1077_two_confusion.png
last/board_1077_two_confusion.png
```

`fivefold_mean_std.csv` and the `Board 1077_two` Excel sheet store values in the
form `87.89% +/- 0.69%` (the file uses the plus-minus symbol). Balanced
accuracy is not calculated or saved.
