import os
import sys
import re
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

def log_results(output_file, message):
    with open(output_file, "a") as file:
        file.write(message + "\n")

def process_file(file_path, weights, threshold_val, output_suffix, platform):
    output_dir = os.path.join('eval', platform, 'results')
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    column_names = ['Battle_Score', 'CatBoost', 'LightGBM', 'MLP', 'SVM', 'XGBoost']

    if not all(col in df.columns for col in column_names):
        log_results(os.path.join(output_dir, 'results.log'), "Required columns not found in the file.")
        return

    df['final_pred_score'] = sum(df[col] * weight for col, weight in zip(column_names, weights))
    df['final_pred'] = df['final_pred_score'].apply(lambda x: 1 if x >= threshold_val else 0)

    output_file = os.path.join(output_dir, f"{output_suffix}_pred.csv")
    df.to_csv(output_file, index=False)
    log_results(os.path.join(output_dir, 'results.log'), f"Prediction file saved at {output_file}")

    tn, fp, fn, tp = confusion_matrix(df['True Label'], df['final_pred']).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    fpr, tpr, _ = roc_curve(df['True Label'], df['final_pred_score'])
    auc_roc = auc(fpr, tpr)

    # Convert all metrics to native Python types (e.g., int, float)
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Accuracy": float(accuracy),
        "F1 Score": float(f1_score),
        "Recall": float(recall),
        "AUC-ROC": float(auc_roc)
    }

    metrics_json = json.dumps(metrics, indent=4)
    log_results(os.path.join(output_dir, 'results.log'), f"Optimal Parameters for {output_suffix}:\n{metrics_json}")

    results_file = os.path.join(output_dir, f"{output_suffix}_results.json")
    with open(results_file, 'w') as f:
        f.write(metrics_json)

    print(f"Metrics saved to {results_file}")

def parse_params_and_run(file_path, text_file, platform):
    with open(text_file, 'r') as file:
        text = file.read()

    pattern = r"Optimal Parameters for (.*?):\nBest .*?Weights: \[(.*?)\]\nBest Threshold: ([0-9.]+)"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        metric, weights_str, threshold = match
        weights_list = [float(w) for w in weights_str.split()]
        process_file(file_path, weights_list, float(threshold), metric.replace(" ", "_").replace(">", "").replace("<", ""), platform)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py PLATFORM")
        sys.exit(1)

    platform = sys.argv[1]
    csv_file_path = os.path.join(platform, 'test', 'AdaptLink_test.csv')
    text_file_path = os.path.join(platform, 'optimize_results.txt')
    
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(text_file_path):
        print(f"File {text_file_path} does not exist.")
        sys.exit(1)

    parse_params_and_run(csv_file_path, text_file_path, platform)
