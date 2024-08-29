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

    with open(output_file, "a") as file:  # Use "a" mode to append to the file
        file.write(message + "\n")

def process_files(platform, output_file):
    val_dir = os.path.join(platform, 'val')
    adaptlink_file = 'AdaptLink_val.csv'
    adaptlink_path = os.path.join(val_dir, adaptlink_file)

    if not os.path.exists(adaptlink_path):
        log_results(output_file, f"File {adaptlink_file} does not exist in {val_dir}.")
        return

    df = pd.read_csv(adaptlink_path)

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

    # Initialize best accuracy when recall > 0.85
    best_accuracy_recall_above_085 = 0
    best_weights_accuracy_recall_above_085 = {}
    best_threshold_accuracy_recall_above_085 = 0

    # Initialize best accuracy when recall > 0.9
    best_accuracy_recall_above_09 = 0
    best_weights_accuracy_recall_above_09 = {}
    best_threshold_accuracy_recall_above_09 = 0

    # Initialize best accuracy when recall > 0.7
    best_accuracy_recall_above_07 = 0
    best_weights_accuracy_recall_above_07 = {}
    best_threshold_accuracy_recall_above_07 = 0

    # Initialize best accuracy when recall > 0.75
    best_accuracy_recall_above_075 = 0
    best_weights_accuracy_recall_above_075 = {}
    best_threshold_accuracy_recall_above_075 = 0

    # Number of random combinations to try
    n_combinations = 10000

    column_names = ['Battle_Score', 'CatBoost', 'LightGBM', 'MLP', 'SVM', 'XGBoost']

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
    
        # Update best scores, weights, thresholds, and metrics
        if f1 > best_f1:
            best_f1 = f1
            best_weights_f1 = weights
            best_threshold_f1 = threshold_val
            metrics_best_f1 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

        if recall > best_recall:
            best_recall = recall
            best_weights_recall = weights
            best_threshold_recall = threshold_val
            metrics_best_recall = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights_accuracy = weights
            best_threshold_accuracy = threshold_val
            metrics_best_accuracy = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_weights_auc_roc = weights
            best_threshold_auc_roc = threshold_val
            metrics_best_auc_roc = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}
    
        if recall > 0.8 and accuracy > best_accuracy_recall_above_08:
            best_accuracy_recall_above_08 = accuracy
            best_weights_accuracy_recall_above_08 = weights
            best_threshold_accuracy_recall_above_08 = threshold_val
            metrics_best_recall_above_08 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

        if recall > 0.85 and accuracy > best_accuracy_recall_above_085:
            best_accuracy_recall_above_085 = accuracy
            best_weights_accuracy_recall_above_085 = weights
            best_threshold_accuracy_recall_above_085 = threshold_val
            metrics_best_recall_above_085 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

        if recall > 0.9 and accuracy > best_accuracy_recall_above_09:
            best_accuracy_recall_above_09 = accuracy
            best_weights_accuracy_recall_above_09 = weights
            best_threshold_accuracy_recall_above_09 = threshold_val
            metrics_best_recall_above_09 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}
    
        if recall > 0.7 and accuracy > best_accuracy_recall_above_07:
            best_accuracy_recall_above_07 = accuracy
            best_weights_accuracy_recall_above_07 = weights
            best_threshold_accuracy_recall_above_07 = threshold_val
            metrics_best_recall_above_07 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}
    
        if recall > 0.75 and accuracy > best_accuracy_recall_above_075:
            best_accuracy_recall_above_075 = accuracy
            best_weights_accuracy_recall_above_075 = weights
            best_threshold_accuracy_recall_above_075 = threshold_val
            metrics_best_recall_above_075 = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'AUC-ROC': auc_roc}

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

    log_results(output_file, "\nOptimal Parameters for Recall:")
    log_results(output_file, f"Best Recall: {best_recall}")
    log_results(output_file, f"Best Weights: {best_weights_recall}")
    log_results(output_file, f"Best Threshold: {best_threshold_recall}")
    log_results(output_file, metrics_best_recall)

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

    log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.7:")
    log_results(output_file, f"Best Accuracy (with Recall > 0.7): {best_accuracy_recall_above_07}")
    log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_07}")
    log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_07}")
    log_results(output_file, metrics_best_recall_above_07)

    log_results(output_file, "\nOptimal Parameters for Accuracy when Recall > 0.75:")
    log_results(output_file, f"Best Accuracy (with Recall > 0.75): {best_accuracy_recall_above_075}")
    log_results(output_file, f"Best Weights: {best_weights_accuracy_recall_above_075}")
    log_results(output_file, f"Best Threshold: {best_threshold_accuracy_recall_above_075}")
    log_results(output_file, metrics_best_recall_above_075)

def main():
    if len(sys.argv) == 1:
        platforms = ['mobile', 'pc']
    elif len(sys.argv) == 2:
        platforms = [sys.argv[1]]
    else:
        print("Usage: python script_name.py [platform]")
        sys.exit(1)
    
    for platform in platforms:
        output_file = os.path.join(platform, 'optimize_results.txt')
        
        os.makedirs(platform, exist_ok=True)
        
        process_files(platform, output_file)

if __name__ == "__main__":
    main()