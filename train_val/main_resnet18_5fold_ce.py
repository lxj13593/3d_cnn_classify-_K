import os
import sys
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

os.environ['MPLBACKEND'] = 'Agg'

# 2. 告诉 Qt 使用离屏渲染，不创建真实窗口
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

qt_platforms_path = Path(sys.prefix) / 'Library' / 'plugins' / 'platforms'
if qt_platforms_path.exists():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_platforms_path)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_operate.data_load_0 import get_data_loaders
from model.resnet_18_34_50_3d import ResNet3D, BasicBlock
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

# 导入统一配置
from data_operate.config import (
    BATCH_SIZE,
    DATA_5FOLD_DIR,
    DISPLAY_LABELS,
    DROPOUT_RATE,
    EPOCHS_PER_RUN,
    FOLD_SEEDS,
    LEARNING_RATE,
    LR_ETA_MIN,
    AMP_ENABLED,
    MODEL_BEST_LAST_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    RAW_DTYPE,
    RESNET18_CE_EXPERIMENT,
    RESNET18_CE_MODEL_FILES,
    TRAIN_VAL_RESULT_DIR,
    USE_DETERMINISTIC_ALGORITHMS,
    USE_GLOBAL_NORM,
    VOLUME_SHAPE,
    WEIGHT_DECAY,
)


def set_seed(seed=RANDOM_SEED):
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(USE_DETERMINISTIC_ALGORITHMS)
    print(f"随机种子已设置: {seed}")


GREEN = "\033[92m"  # 绿色
BLUE = "\033[94m"  # 蓝色
YELLOW = "\033[93m"  # 黄色
RED = "\033[91m"  # 红色
RESET = "\033[0m"  # 重置


# ================= 3. 训练与测试流程 =================
def run_pipeline(fold_idx, seed, num_epochs=EPOCHS_PER_RUN):
    """运行训练流程"""
    # 设置随机种子
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 使用 ResNet18 + Dropout正则化
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

    data_root = str(DATA_5FOLD_DIR / f'fold_{fold_idx}')
    print(f"\n数据根目录: {data_root}")

    train_loader, val_loader = get_data_loaders(
        data_root=data_root,
        shape=VOLUME_SHAPE,
        dtype=RAW_DTYPE,
        batch_size=BATCH_SIZE,  # 根据显存大小调整，如OOM可改为2或1
        num_workers=NUM_WORKERS,
        seed=seed,  # 使用当前轮次的随机种子
        use_global_norm=USE_GLOBAL_NORM  # 使用全局归一化
    )

    # --- 修改：按种子隔离输出路径，防止多轮相互覆盖 ---
    model_dir = str(MODEL_BEST_LAST_DIR / RESNET18_CE_EXPERIMENT / f'fold_{fold_idx}')
    picture_dir = str(TRAIN_VAL_RESULT_DIR / RESNET18_CE_EXPERIMENT / f'fold_{fold_idx}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)

    # --- 使用标准的交叉熵损失函数 ---
    criterion = nn.CrossEntropyLoss()
    print("使用 Cross Entropy Loss")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率调度器 - CosineAnnealingLR 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR_ETA_MIN)

    amp_enabled = AMP_ENABLED and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    print(f"启用 AMP 混合精度加速: {amp_enabled}")

    best_val_f1 = -1.0
    best_val_loss = float('inf')  # 新增：初始化最低验证损失为正无穷

    # 用于多轮实验汇总的最优指标记录器（仅保存最稳定的基础指标，防止中途崩溃）
    best_metrics_summary = {'val_f1': 0.0, 'val_acc': 0.0}

    # 记录训练历史 (用于画曲线图)
    train_losses, train_accs, val_losses, val_accs, learning_rates = [], [], [], [], []
    train_metrics, val_metrics = [], []  # [(precision, recall, f1, specificity), ...]
    train_losses_running = []  # 记录训练过程中的 running loss（train 模式下）

    print(f"\n训练集: {len(train_loader.dataset)}")
    print(f"验证集: {len(val_loader.dataset)}")
    print("\n开始训练...\n" + "=" * 60)

    for epoch in range(num_epochs):
        # ===== train =====
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds, all_train_labels = [], []  # 收集训练集预测结果

        with tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Train', file=sys.stdout,
                  dynamic_ncols=True) as tepoch:
            for batch_idx, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # 收集预测结果用于计算其他指标
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                tepoch.set_postfix(loss=f'{running_loss / (batch_idx + 1):.4f}')

        # 记录训练过程中的 running loss（train 模式下的实际损失）
        avg_train_loss_running = running_loss / len(train_loader)
        train_losses_running.append(avg_train_loss_running)

        # 计算训练集准确率和所有指标
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss_running)
        train_accs.append(train_acc)

        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        cm_train = confusion_matrix(all_train_labels, all_train_preds, labels=[0, 1])
        train_specificity = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1]) if (cm_train[0, 0] + cm_train[
            0, 1]) > 0 else 0.0
        train_metrics.append((train_precision, train_recall, train_f1, train_specificity))

        # ===== val =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds_list, all_val_labels_list, all_val_probs_list = [], [], []

        with tqdm(val_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Val', file=sys.stdout,
                  dynamic_ncols=True) as vepoch:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(vepoch):
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()

                    # 获取正类的预测概率
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # 收集预测结果用于计算其他指标
                    all_val_preds_list.extend(predicted.cpu().numpy())
                    all_val_labels_list.extend(labels.cpu().numpy())
                    all_val_probs_list.extend(probs.cpu().numpy())

                    vepoch.set_postfix(loss=f'{val_loss / (batch_idx + 1):.4f}')

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # 计算验证集其他指标
        val_precision = precision_score(all_val_labels_list, all_val_preds_list, zero_division=0)
        val_recall = recall_score(all_val_labels_list, all_val_preds_list, zero_division=0)
        val_f1 = f1_score(all_val_labels_list, all_val_preds_list, zero_division=0)
        cm_val = confusion_matrix(all_val_labels_list, all_val_preds_list, labels=[0, 1])
        val_specificity = cm_val[0, 0] / (cm_val[0, 0] + cm_val[0, 1]) if (cm_val[0, 0] + cm_val[0, 1]) > 0 else 0.0
        val_metrics.append((val_precision, val_recall, val_f1, val_specificity))

        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        scheduler.step()

        # 当验证集 F1 创新高时，保存最佳模型并生成图表
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            # 保存最佳模型
            save_path = os.path.join(model_dir, RESNET18_CE_MODEL_FILES['best_f1'])
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': best_val_f1,
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'learning_rate': current_lr,
                'norm_params': {
                    'global_min': train_loader.dataset.global_min,
                    'global_max': train_loader.dataset.global_max
                }
            }, save_path)
            print(f"🎉 新高! 最佳F1模型已保存: {save_path} (F1: {best_val_f1:.4f}, Acc: {val_acc:.2f}%)")

            # 生成混淆矩阵和评估曲线
            print(f"基于最佳模型生成混淆矩阵和 AUC 曲线...")

            # 混淆矩阵
            cm = confusion_matrix(all_val_labels_list, all_val_preds_list, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(DISPLAY_LABELS))
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix (Best Epoch {epoch + 1}, F1: {best_val_f1:.4f}, Acc: {val_acc:.2f}%)')
            cm_path = os.path.join(picture_dir, 'confusion_matrix_best.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"混淆矩阵已保存: {cm_path}")

            # AUC 曲线（复用验证集概率）
            fpr, tpr, _ = roc_curve(all_val_labels_list, all_val_probs_list)
            best_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (Best Epoch {epoch + 1}, AUC: {best_auc:.4f})')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            auc_best_path = os.path.join(picture_dir, 'auc_best.png')
            plt.savefig(auc_best_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"最佳模型 AUC 曲线已保存: {auc_best_path}, AUC: {best_auc:.4f}")

            # PR 曲线
            precision_vals, recall_vals, _ = precision_recall_curve(all_val_labels_list, all_val_probs_list)
            best_ap = average_precision_score(all_val_labels_list, all_val_probs_list)

            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='darkblue', lw=2, label=f'PR curve (AP = {best_ap:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (Best Epoch {epoch + 1}, AP: {best_ap:.4f})')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            pr_best_path = os.path.join(picture_dir, 'pr_curve_best.png')
            plt.savefig(pr_best_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"最佳模型 PR 曲线已保存: {pr_best_path}, AP: {best_ap:.4f}")

            # 更新当前种子的最优核心表现
            best_metrics_summary = {
                'val_f1': val_f1,
                'val_acc': val_acc
            }

        # --- 当验证集 Loss 创新低时，保存最低 Loss 模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_loss_path = os.path.join(model_dir, RESNET18_CE_MODEL_FILES['best_loss'])
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'learning_rate': current_lr,
                'norm_params': {
                    'global_min': train_loader.dataset.global_min,
                    'global_max': train_loader.dataset.global_max
                }
            }, save_loss_path)
            print(f"📉 损创新低! 最低 Loss 模型已保存: {save_loss_path} (Loss: {best_val_loss:.4f})")

        # 打印进度
        summary_line1 = (
            f"{GREEN}RESULT: Epoch [{epoch + 1:02d}]{RESET} | "
            f"{BLUE}Train Loss: {avg_train_loss_running:.4f}{RESET} | "
            f"{YELLOW}Val Loss: {avg_val_loss:.4f}{RESET} | "
            f"{RED}LR: {current_lr:.6f}{RESET}"
        )
        summary_line2 = (
            f"{BLUE}Train - A: {train_acc:.2f}% P: {train_precision:.4f} R: {train_recall:.4f} F1: {train_f1:.4f} TNR: {train_specificity:.4f}{RESET} | "
            f"{YELLOW}Val   - A: {val_acc:.2f}% P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f} TNR: {val_specificity:.4f}{RESET}"
        )
        print(summary_line1)
        print(summary_line2)
        print("-" * 100)

    # ================= 绘制图表 =================
    print("\n训练结束，正在生成最终报告...")

    # 5. 保存最后一轮模型
    last_model_path = os.path.join(model_dir, RESNET18_CE_MODEL_FILES['last'])
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_f1': val_f1,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc,
        'norm_params': {
            'global_min': train_loader.dataset.global_min,
            'global_max': train_loader.dataset.global_max
        }
    }, last_model_path)
    print(f"最后一轮模型已保存: {last_model_path}")

    # 基于最后一轮模型生成最终评估图表
    print("使用最后一轮模型进行最终独立评估...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    # 最终模型混淆矩阵
    cm_last = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_last, display_labels=list(DISPLAY_LABELS))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (Last Epoch {num_epochs})')
    cm_last_path = os.path.join(picture_dir, 'confusion_matrix_last.png')
    plt.savefig(cm_last_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"最终模型混淆矩阵已保存: {cm_last_path}")

    # 最终模型 AUC 曲线
    fpr_last, tpr_last, _ = roc_curve(all_labels, all_probs)
    auc_last = auc(fpr_last, tpr_last)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_last, tpr_last, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_last:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Last Epoch {num_epochs}, AUC: {auc_last:.4f})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    auc_last_path = os.path.join(picture_dir, 'auc_last.png')
    plt.savefig(auc_last_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最终模型 AUC 曲线已保存: {auc_last_path}, AUC: {auc_last:.4f}")

    # 最终模型 PR 曲线
    precision_vals_last, recall_vals_last, _ = precision_recall_curve(all_labels, all_probs)
    ap_last = average_precision_score(all_labels, all_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals_last, precision_vals_last, color='darkblue', lw=2, label=f'PR curve (AP = {ap_last:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Last Epoch {num_epochs}, AP: {ap_last:.4f})')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    pr_last_path = os.path.join(picture_dir, 'pr_curve_last.png')
    plt.savefig(pr_last_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最终模型 PR 曲线已保存: {pr_last_path}, AP: {ap_last:.4f}")

    # 1. 绘制损失曲线（仅包含 running loss 和 val loss）
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses_running, 'b--', label='Train Loss (running, train mode)',
             linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(picture_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {loss_path}")

    # 2. 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accs, 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    acc_path = os.path.join(picture_dir, 'accuracy_curve.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2.1 绘制从50%开始的准确率曲线（放大细节）
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_accs, 'r-', label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy (Zoomed from 50%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(50, 100)  # 从50%开始显示
    acc_from_50_path = os.path.join(picture_dir, 'accuracy_curve_from_50.png')
    plt.savefig(acc_from_50_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"准确率曲线已保存: {acc_path}")
    print(f"从50%开始的准确率曲线已保存: {acc_from_50_path}")

    # 2.2 绘制F1分数曲线
    train_f1s = [m[2] for m in train_metrics]
    val_f1s = [m[2] for m in val_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_f1s, 'b-', label='Train F1-Score', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_f1s, 'r-', label='Val F1-Score', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training and Validation F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    f1_path = os.path.join(picture_dir, 'f1_curve.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1曲线已保存: {f1_path}")

    # 2.3 绘制从0.5开始的F1分数曲线（放大细节）
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_f1s, 'b-', label='Train F1-Score', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_f1s, 'r-', label='Val F1-Score', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training and Validation F1-Score (Zoomed from 0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)  # 从0.5开始显示
    f1_from_05_path = os.path.join(picture_dir, 'f1_curve_from_05.png')
    plt.savefig(f1_from_05_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"从0.5开始的F1曲线已保存: {f1_from_05_path}")

    # 3. 绘制学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), learning_rates, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Cosine Annealing)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 对数坐标更好地显示学习率变化
    lr_path = os.path.join(picture_dir, 'learning_rate_curve.png')
    plt.savefig(lr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学习率曲线已保存: {lr_path}")

    # 6. 保存所有指标到 Excel
    metrics_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train_Loss': train_losses_running,
        'Train_Accuracy': train_accs,
        'Train_Precision': [m[0] for m in train_metrics],
        'Train_Recall': [m[1] for m in train_metrics],
        'Train_F1': [m[2] for m in train_metrics],
        'Train_Specificity': [m[3] for m in train_metrics],
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accs,
        'Val_Precision': [m[0] for m in val_metrics],
        'Val_Recall': [m[1] for m in val_metrics],
        'Val_F1': [m[2] for m in val_metrics],
        'Val_Specificity': [m[3] for m in val_metrics],
        'Learning_Rate': learning_rates
    })
    excel_path = os.path.join(picture_dir, 'training_metrics.xlsx')
    metrics_df.to_excel(excel_path, index=False)
    print(f"训练指标已保存到 Excel: {excel_path}")

    return best_metrics_summary


# ================= 主程序入口：执行 5-fold 循环 =================
if __name__ == '__main__':
    # 5-fold × 1 seed。关键：同一个 fold 内，不同模型建议使用相同 seed。
    fold_seeds = FOLD_SEEDS

    all_fold_results = []

    print("=" * 70)
    print(f" 开始执行 ResNet18 + CE 5-fold 训练。fold seeds: {fold_seeds} | 单折: {EPOCHS_PER_RUN} 轮")
    print("=" * 70)

    for fold_idx, seed in enumerate(fold_seeds):
        print(f"\n{'#' * 15} 正在运行 Fold {fold_idx}/4 (Seed: {seed}) {'#' * 15}")
        run_summary = run_pipeline(fold_idx=fold_idx, seed=seed, num_epochs=EPOCHS_PER_RUN)
        run_summary['fold'] = fold_idx
        run_summary['seed'] = seed
        all_fold_results.append(run_summary)
        print(f"Fold {fold_idx} 完成。本折最高验证集 F1: {run_summary['val_f1']:.4f}")

    df_results = pd.DataFrame(all_fold_results)
    df_results.index = [f"Fold_{i}" for i in range(len(all_fold_results))]

    means = df_results.select_dtypes(include=[np.number]).mean()
    stds = df_results.select_dtypes(include=[np.number]).std()

    summary_root = TRAIN_VAL_RESULT_DIR / RESNET18_CE_EXPERIMENT
    summary_root.mkdir(parents=True, exist_ok=True)
    raw_path = summary_root / 'fivefold_val_summary_raw.xlsx'
    df_results.to_excel(raw_path, index=True)

    print("\n" + "=" * 25 + "  5-fold 验证集表现统计  " + "=" * 25)
    print(df_results.to_string())
    print("-" * 80)
    print("验证集综合性能指标 (Mean ± Std):")
    if 'val_acc' in means:
        print(f"  • Val Accuracy  : {means['val_acc']:.2f}% ± {stds['val_acc']:.2f}%")
    if 'val_f1' in means:
        print(f"  • Val F1-Score  : {means['val_f1']:.4f} ± {stds['val_f1']:.4f}")
    print(f"原始验证集汇总已保存: {raw_path}")
    print("=" * 80)
