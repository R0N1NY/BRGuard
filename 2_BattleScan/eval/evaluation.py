import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, recall_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
                             roc_curve, auc, precision_recall_curve)

pt = sys.argv[1]

def process_stats_and_plots(mode):
    output_dir = os.path.join(os.getcwd(), pt)
    os.chdir(output_dir)
    
    df = pd.read_csv(f'BattleScan_{mode}.csv')
    y_true = df['True Label']
    y_pred = df['Prediction']

    results = pd.DataFrame(columns=['Metric', 'Value'])

    # 计算各项指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = recall_score(y_true, y_pred)  # TPR
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    accuracy = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    npv = tn / (tn + fn)
    N = len(y_true)
    oei = (N / (tp + fp)) * (recall * npv)

    # 将结果存入DataFrame
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': accuracy,
        'Recall (TPR)': recall,
        'NPV': npv,
        'FPR': fpr,
        'AUC-ROC': auc_roc,
        'Operational Efficiency Index': oei,
        'FNR': fnr
    }

    for metric_name, metric_value in metrics.items():
        results = pd.concat([results, pd.DataFrame({'Metric': [metric_name], 'Value': [metric_value]})], ignore_index=True)

    results.to_csv(f'BattleScan_{mode}_results.csv', index=False)

    # 绘制混淆矩阵、ROC曲线和Precision-Recall曲线
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].fill_between(fpr, tpr, color='darkorange', alpha=0.1)
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")

    ax[2].step(recall, precision, color='b', alpha=0.2, where='post')
    ax[2].fill_between(recall, precision, alpha=0.2, color='b')
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_ylim([0.0, 1.05])
    ax[2].set_xlim([0.0, 1.0])

    plt.tight_layout()
    plt.savefig(f'BattleScan_{mode}.pdf')

    os.chdir('..')
    print(f"Results for {mode.upper()} mode:")
    print(results)

def main():
    modes = ['test', 'val'] if len(sys.argv) == 2 else [sys.argv[2]]
    for mode in modes:
        process_stats_and_plots(mode)

if __name__ == '__main__':
    main()
