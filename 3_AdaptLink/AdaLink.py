import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm
import json

def log_results(output_file, message):
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    if isinstance(message, dict):
        message = json.dumps(message, indent=4, default=default)

    with open(output_file, "a") as file:
        file.write(message + "\n")

def process_files(platform, threshold_option):
    val_dir = os.path.join(platform, 'val')
    adaptlink_file = 'AdaptLink_val.csv'
    adaptlink_path = os.path.join(val_dir, adaptlink_file)

    if not os.path.exists(adaptlink_path):
        log_results(output_file, f"File {adaptlink_file} does not exist in {val_dir}.")
        return

    df = pd.read_csv(adaptlink_path)

    column_names = ['Battle_Score', 'CatBoost', 'LightGBM', 'MLP', 'SVM', 'XGBoost']

    if threshold_option == 'nofalse':
        output_file = os.path.join(platform, 'optimize_results_nofalse.txt')

        weights = np.ones(len(column_names)) 
        threshold_val = 5.5 

        df['final_pred_score'] = sum(df[col] * weight for col, weight in zip(column_names, weights))
        df['final_pred'] = df['final_pred_score'].apply(lambda x: 1 if x >= threshold_val else 0)

        tn, fp, fn, tp = confusion_matrix(df['True Label'], df['final_pred']).ravel()
        accuracy = accuracy_score(df['True Label'], df['final_pred'])
        f1 = f1_score(df['True Label'], df['final_pred'])
        recall = recall_score(df['True Label'], df['final_pred'])
        auc_roc = roc_auc_score(df['True Label'], df['final_pred'])
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)

        metrics_best_fpr = {
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall,
            'AUC-ROC': auc_roc,
            'FNR': fnr,
            'FPR': fpr
        }

        log_results(output_file, "\nOptimal Parameters for nofalse (fixed weights and threshold):")
        log_results(output_file, f"Fixed Weights: {' '.join(map(str, weights))}")
        log_results(output_file, f"Fixed Threshold: {threshold_val}")
        log_results(output_file, metrics_best_fpr)

    elif threshold_option == 'nomiss':
        output_file = os.path.join(platform, 'optimize_results_nomiss.txt')
        best_fnr = float('inf')
        best_weights_fnr = {}
        best_threshold_fnr = 0
        metrics_best_fnr = {}

        for _ in tqdm(range(10000), desc="Searching for nomiss (low FNR)"):
            weights = np.random.dirichlet(np.ones(len(column_names)), size=1)[0]
            weight_dict = dict(zip(column_names, weights))

            df['final_pred_score'] = sum(df[col] * weight for col, weight in weight_dict.items())
            threshold_val = np.random.uniform(0, 1)
            df['final_pred'] = df['final_pred_score'].apply(lambda x: 1 if x >= threshold_val else 0)
            recall = recall_score(df['True Label'], df['final_pred'])
            accuracy = accuracy_score(df['True Label'], df['final_pred'])
            tn, fp, fn, tp = confusion_matrix(df['True Label'], df['final_pred']).ravel()
            fnr = fn / (fn + tp)

            if fnr < best_fnr and accuracy > 0.85:
                best_fnr = fnr
                best_weights_fnr = weights
                best_threshold_fnr = threshold_val
                metrics_best_fnr = {
                    'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                    'Accuracy': accuracy_score(df['True Label'], df['final_pred']),
                    'F1 Score': f1_score(df['True Label'], df['final_pred']),
                    'Recall': recall_score(df['True Label'], df['final_pred']),
                    'AUC-ROC': roc_auc_score(df['True Label'], df['final_pred']),
                    'FNR': fnr, 'FPR': fp / (fp + tn)
                }

        log_results(output_file, "\nOptimal Parameters for nomiss (lowest FNR):")
        log_results(output_file, f"Best Weights: {best_weights_fnr}")
        log_results(output_file, f"Best Threshold: {best_threshold_fnr}")
        log_results(output_file, metrics_best_fnr)


    else:
        output_file = os.path.join(platform, 'optimize_results.txt')

        # Initialize best scores
        best_f1 = 0
        best_recall = 0
        best_accuracy = 0
        best_auc_roc = 0

        best_weights_f1 = {}
        best_weights_recall = {}
        best_weights_accuracy = {}
        best_weights_auc_roc = {}

        best_threshold_f1 = 0
        best_threshold_recall = 0
        best_threshold_accuracy = 0
        best_threshold_auc_roc = 0

        # Metrics for best parameters
        metrics_best_f1 = {}
        metrics_best_recall = {}
        metrics_best_accuracy = {}
        metrics_best_auc_roc = {}

        # Initialize best accuracy when recall > 0.8
        best_accuracy_recall_above_08 = 0
        best_weights_accuracy_recall_above_08 = {}
        best_threshold_accuracy_recall_above_08 = 0
        metrics_best_recall_above_08 = {}

        # Initialize best accuracy when recall > 0.85
        best_accuracy_recall_above_085 = 0
        best_weights_accuracy_recall_above_085 = {}
        best_threshold_accuracy_recall_above_085 = 0
        metrics_best_recall_above_085 = {}

        # Initialize best accuracy when recall > 0.9
        best_accuracy_recall_above_09 = 0
        best_weights_accuracy_recall_above_09 = {}
        best_threshold_accuracy_recall_above_09 = 0
        metrics_best_recall_above_09 = {}

        # Initialize best accuracy when recall > 0.95
        best_accuracy_recall_above_095 = 0
        best_weights_accuracy_recall_above_095 = {}
        best_threshold_accuracy_recall_above_095 = 0
        metrics_best_recall_above_095 = {}

        # Number of random combinations to try
        n_combinations = 10000

        for _ in tqdm(range(n_combinations), desc="Searching"):

            weights = np.random.dirichlet(np.ones(len(column_names)), size=1)[0]
            
            weight_dict = dict(zip(column_names, weights))
            
            df['final_pred_score'] = sum(df[col] * weight for col, weight in weight_dict.items())
            threshold_val = np.random.uniform(0, 1)

            df['final_pred'] = df['final_pred_score'].apply(lambda x: 1 if x >= threshold_val else 0)

            accuracy = accuracy_score(df['True Label'], df['final_pred'])
            f1 = f1_score(df['True Label'], df['final_pred'])
            recall = recall_score(df['True Label'], df['final_pred'])
            auc_roc = roc_auc_score(df['True Label'], df['final_pred'])
            tn, fp, fn, tp = confusion_matrix(df['True Label'], df['final_pred']).ravel()
            fnr = fn / (fn + tp)
            fpr = fp / (fp + tn)

            # Update best scores, weights, thresholds, and metrics
            if f1 > best_f1:
                best_f1 = f1
                best_weights_f1 = weights
                best_threshold_f1 = threshold_val
                metrics_best_f1 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights_accuracy = weights
                best_threshold_accuracy = threshold_val
                metrics_best_accuracy = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_weights_auc_roc = weights
                best_threshold_auc_roc = threshold_val
                metrics_best_auc_roc = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}
        
            if recall > 0.8 and accuracy > best_accuracy_recall_above_08:
                best_accuracy_recall_above_08 = accuracy
                best_weights_accuracy_recall_above_08 = weights
                best_threshold_accuracy_recall_above_08 = threshold_val
                metrics_best_recall_above_08 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

            if recall > 0.85 and accuracy > best_accuracy_recall_above_085:
                best_accuracy_recall_above_085 = accuracy
                best_weights_accuracy_recall_above_085 = weights
                best_threshold_accuracy_recall_above_085 = threshold_val
                metrics_best_recall_above_085 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

            if recall > 0.9 and accuracy > best_accuracy_recall_above_09:
                best_accuracy_recall_above_09 = accuracy
                best_weights_accuracy_recall_above_09 = weights
                best_threshold_accuracy_recall_above_09 = threshold_val
                metrics_best_recall_above_09 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

            if recall > 0.95 and accuracy > best_accuracy_recall_above_095:
                best_accuracy_recall_above_095 = accuracy
                best_weights_accuracy_recall_above_095 = weights
                best_threshold_accuracy_recall_above_095 = threshold_val
                metrics_best_recall_above_095 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc, 'FNR': fnr, 'FPR': fpr}

        # Log results
        log_results(output_file, "\nOptimal Parameters for F1:")
        log_results(output_file, f"Best F1 Score: {best_f1}")
        log_results(output_file, f"Best Weights: {best_weights_f1}")
        log_results(output_file, f"Best Threshold: {best_threshold_f1}")
        log_results(output_file, metrics_best_f1)

        log_results(output_file, "\nOptimal Parameters for AUC-ROC:")
        log_results(output_file, f"Best Accuracy: {best_auc_roc}")
        log_results(output_file, f"Best Weights: {best_weights_auc_roc}")
        log_results(output_file, f"Best Threshold: {best_threshold_auc_roc}")
        log_results(output_file, metrics_best_auc_roc)

        log_results(output_file, "\nOptimal Parameters for Accuracy:")
        log_results(output_file, f"Best Accuracy: {best_accuracy}")
        log_results(output_file, f"Best Weights: {best_weights_accuracy}")
        log_results(output_file, f"Best Threshold: {best_threshold_accuracy}")
        log_results(output_file, metrics_best_accuracy)

        log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.8:")
        log_results(output_file, f"Best Accuracy (with Recall > 0.8): {best_accuracy_recall_above_08}")
        log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_08}")
        log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_08}")
        log_results(output_file, metrics_best_recall_above_08)

        log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.85:")
        log_results(output_file, f"Best Accuracy (with Recall > 0.85): {best_accuracy_recall_above_085}")
        log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_085}")
        log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_085}")
        log_results(output_file, metrics_best_recall_above_085)

        log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.9:")
        log_results(output_file, f"Best Accuracy (with Recall > 0.9): {best_accuracy_recall_above_09}")
        log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_09}")
        log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_09}")
        log_results(output_file, metrics_best_recall_above_09)

        log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.95:")
        log_results(output_file, f"Best Accuracy (with Recall > 0.95): {best_accuracy_recall_above_095}")
        log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_095}")
        log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_095}")
        log_results(output_file, metrics_best_recall_above_095)

def main(platform, threshold_option=None):
    if threshold_option is None:
        threshold_option = 'random'
    process_files(platform, threshold_option)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AdaLink.py <platform> [<threshold_option>]")
    else:
        platform = sys.argv[1]
        threshold_option = sys.argv[2] if len(sys.argv) > 2 else None
        main(platform, threshold_option)
