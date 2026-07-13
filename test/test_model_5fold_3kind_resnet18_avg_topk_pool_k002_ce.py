import os

os.environ['MPLBACKEND'] = 'Agg'

# 2. 告诉 Qt 使用离屏渲染，不创建真实窗口
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import pandas as pd

# 添加父目录到路径，以便导入 train_val 模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 导入统一配置
from data_operate.config import (
    BATCH_SIZE,
    DATA_5FOLD_DIR,
    DISPLAY_LABELS,
    DROPOUT_RATE,
    FOLD_SEEDS,
    MODEL_BEST_LAST_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    RAW_DTYPE,
    RESNET18_AVG_TOPK_POOL_EXPERIMENT,
    RESNET18_AVG_TOPK_POOL_MODEL_FILES,
    TEST_RESULT_DIR,
    TEST_THRESHOLD,
    USE_DETERMINISTIC_ALGORITHMS,
    VOLUME_SHAPE,
)

# 导入自定义模块
from data_operate.data_load_0 import RawCTDataset, compute_global_stats
from model.resnet_18_34_50_3d_avg_topk_pool_k002 import ResNet3D, BasicBlock
from torch.utils.data import DataLoader

RESNET18_AVG_TOPK_POOL_EXPERIMENT = "resnet18_avg_topk_pool_k002_5fold"
RESNET18_AVG_TOPK_POOL_MODEL_FILES = {
    "best_f1": "best_resnet18_avg_topk_pool_k002_3d.pth",
    "best_loss": "best_loss_resnet18_avg_topk_pool_k002_3d.pth",
    "last": "last_resnet18_avg_topk_pool_k002_3d.pth",
}


def set_seed(seed=RANDOM_SEED):
    """设置随机种子，确保实验可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(USE_DETERMINISTIC_ALGORITHMS)
    print(f"随机种子已设置: {seed}")


def get_test_loader(data_root, shape=VOLUME_SHAPE, dtype=RAW_DTYPE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    global_min=None, global_max=None):
    """创建测试集的 DataLoader"""
    test_dir = os.path.join(data_root, 'test')

    # 如果没有提供全局归一化参数，则计算
    if global_min is None or global_max is None:
        print("\n未提供归一化参数，重新计算全局归一化统计量...")
        global_min, global_max = compute_global_stats(data_root, shape, dtype)
        print(f"使用全局归一化: min={global_min}, max={global_max}")
    else:
        print(f"\n使用模型中保存的归一化参数: min={global_min}, max={global_max}")

    # 创建测试数据集（关闭增强）
    test_dataset = RawCTDataset(test_dir, shape, dtype, augment=False,
                                global_min=global_min, global_max=global_max)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def evaluate_model(model, test_loader, device, save_dir=None, threshold=TEST_THRESHOLD):
    """
    评估模型在测试集上的性能
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"\n开始在测试集上评估模型...")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"分类阈值: {threshold}")

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', file=sys.stdout):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 获取预测结果和概率
            probs = torch.softmax(outputs, dim=1)
            prob_defective = probs[:, 1].cpu().numpy()  # 缺陷类别的概率

            # 使用自定义阈值进行分类
            predicted = (prob_defective >= threshold).astype(int)

            all_preds.extend(predicted)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(prob_defective)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算各项指标
    accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 计算AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    # 计算PR曲线和AP
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)

    # 打印结果
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy):  {accuracy:.2f}%")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print(f"特异度 (Specificity): {specificity:.4f}")
    print(f"AUC-ROC:            {auc_score:.4f}")
    print(f"AP (Average Precision): {ap_score:.4f}")
    print("-" * 60)
    print("混淆矩阵:")
    print(f"                预测正常  预测缺陷")
    print(f"实际正常 (TN):    {tn:6d}     {fp:6d}")
    print(f"实际缺陷 (FN):    {fn:6d}     {tp:6d}")
    print("=" * 60)

    # 保存结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc_score,
        'ap': ap_score,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 1. 保存混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(DISPLAY_LABELS))
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix (Test Set, Acc: {accuracy:.2f}%)')
        cm_path = os.path.join(save_dir, 'test_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n混淆矩阵已保存: {cm_path}")

        # 2. 保存ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Test Set, AUC: {auc_score:.4f})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(save_dir, 'test_roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存: {roc_path}")

        # 3. 保存PR曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='darkblue', lw=2,
                 label=f'PR curve (AP = {ap_score:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Test Set, AP: {ap_score:.4f})')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        pr_path = os.path.join(save_dir, 'test_pr_curve.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR曲线已保存: {pr_path}")

        # 4. 保存详细结果到Excel
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score',
                       'Specificity', 'AUC-ROC', 'AP'],
            'Value': [accuracy, precision, recall, f1, specificity, auc_score, ap_score]
        })
        excel_path = os.path.join(save_dir, 'test_metrics.xlsx')
        metrics_df.to_excel(excel_path, index=False)
        print(f"评估指标已保存到: {excel_path}")

        # 4.1 绘制准确率指标对比图（从50%基准线开始）
        plt.figure(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metrics_values = [accuracy / 100, precision, recall, f1, specificity]  # 转换为0-1范围
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        bars = plt.bar(range(len(metrics_names)), metrics_values, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Baseline (50%)')
        plt.xticks(range(len(metrics_names)), metrics_names, rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title(f'Test Set Performance Metrics (Accuracy: {accuracy:.2f}%)')
        plt.ylim(0.4, 1.0)  # 从40%开始显示，突出50%基准线以上的表现
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        metrics_bar_path = os.path.join(save_dir, 'test_metrics_comparison.png')
        plt.savefig(metrics_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标对比图已保存: {metrics_bar_path}")

        # 5. 保存每个样本的预测结果
        predictions_df = pd.DataFrame({
            'Sample_Index': range(len(all_labels)),
            'True_Label': all_labels,
            'Predicted_Label': all_preds,
            'Defective_Probability': all_probs,
            'Normal_Probability': 1 - all_probs,
            'Correct': all_preds == all_labels
        })
        pred_path = os.path.join(save_dir, 'test_predictions.xlsx')
        predictions_df.to_excel(pred_path, index=False)
        print(f"预测详情已保存到: {pred_path}")

    return results


def main():
    """5-fold 测试：每折读取对应 fold 的 test 集，并评估 best_f1 / best_loss / last。"""
    fold_seeds = FOLD_SEEDS
    model_files = RESNET18_AVG_TOPK_POOL_MODEL_FILES

    all_test_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print("=" * 80)
    print("开始执行 ResNet18 avg-topk-pool + CE 5-fold 测试")
    print("=" * 80)

    for fold_idx, seed in enumerate(fold_seeds):
        data_root = str(DATA_5FOLD_DIR / f'fold_{fold_idx}')

        for model_key, file_name in model_files.items():
            print("\n" + "#" * 30)
            print(f"  正在测试: Fold {fold_idx}/4 | Seed {seed} | 模型类型 [{model_key}]")
            print("#" * 30)

            set_seed(seed)

            model_path = str(MODEL_BEST_LAST_DIR / RESNET18_AVG_TOPK_POOL_EXPERIMENT / f'fold_{fold_idx}' / file_name)
            save_dir = str(TEST_RESULT_DIR / RESNET18_AVG_TOPK_POOL_EXPERIMENT / f'fold_{fold_idx}' / model_key)

            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在: {model_path}，已跳过。")
                continue

            print(f"加载模型: {model_path}")
            print(f"测试数据: {data_root}")
            print(f"图表保存: {save_dir}")

            model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
            model = model.to(device)

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载成功!")

            global_min = None
            global_max = None
            if 'norm_params' in checkpoint:
                norm_params = checkpoint['norm_params']
                global_min = norm_params.get('global_min')
                global_max = norm_params.get('global_max')
                print(f"提取到当前 fold 训练集归一化参数: min={global_min}, max={global_max}")
            else:
                print("警告: 权重未包含归一化参数，将从当前 fold 的 train 重新计算")

            test_loader = get_test_loader(
                data_root=data_root,
                shape=VOLUME_SHAPE,
                dtype=RAW_DTYPE,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                global_min=global_min,
                global_max=global_max
            )

            results = evaluate_model(model, test_loader, device, save_dir=save_dir, threshold=TEST_THRESHOLD)

            all_test_results.append({
                'Fold': f"Fold_{fold_idx}",
                'Seed': seed,
                'Model_Type': model_key,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'Specificity': results['specificity'],
                'AUC-ROC': results['auc'],
                'AP': results['ap']
            })

    if all_test_results:
        df_all = pd.DataFrame(all_test_results)

        print("\n" + "=" * 35 + "  5-fold 测试集表现一览表  " + "=" * 35)
        print(df_all.to_string(index=False))
        print("=" * 100)

        summary_root = str(TEST_RESULT_DIR / RESNET18_AVG_TOPK_POOL_EXPERIMENT)
        os.makedirs(summary_root, exist_ok=True)

        raw_excel_path = os.path.join(summary_root, 'fivefold_models_test_summary.xlsx')
        df_all.to_excel(raw_excel_path, index=False)
        print(f"表格 1（每折细分成绩）已保存至: {raw_excel_path}")

        metrics_to_calc = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC', 'AP']
        df_pct = df_all.copy()
        for col in metrics_to_calc:
            if col != 'Accuracy':
                df_pct[col] = df_pct[col] * 100

        grouped_stats = df_pct.groupby('Model_Type')[metrics_to_calc].agg(['mean', 'std'])
        summary_rows = []
        for model_key in model_files:
            if model_key not in grouped_stats.index:
                continue
            row_dict = {'Model_Type': model_key}
            for col in metrics_to_calc:
                mean_v = grouped_stats.loc[model_key, (col, 'mean')]
                std_v = grouped_stats.loc[model_key, (col, 'std')]
                row_dict[col] = f"{mean_v:.2f}% ± {std_v:.2f}%"
            summary_rows.append(row_dict)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.set_index('Model_Type', inplace=True)
        df_transposed = df_summary.T.reset_index()
        df_transposed.rename(columns={'index': 'Metric'}, inplace=True)

        summary_excel_path = os.path.join(summary_root, 'academic_summary_mean_std_5fold.xlsx')
        df_transposed.to_excel(summary_excel_path, index=False)
        print(f"表格 2（5-fold Mean ± Std 学术汇总表）已保存至: {summary_excel_path}")
    else:
        print("\n⚠️ 未能成功评估任何模型，请核对训练生成的 `.pth` 文件是否存在。")


if __name__ == '__main__':
    main()
