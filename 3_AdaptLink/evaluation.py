import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

def log_results(output_file, message):
    with open(output_file, "a") as file:
        file.write(message + "\n")

def process_file(platform, weights, threshold_val):
    test_dir = os.path.join(platform, 'test')
    adaptlink_file = 'AdaptLink_test.csv'
    adaptlink_path = os.path.join(test_dir, adaptlink_file)

    if not os.path.exists(adaptlink_path):
        log_results(os.path.join('eval', platform, 'results.log'), f"File {adaptlink_file} does not exist in {test_dir}.")
        return

    df = pd.read_csv(adaptlink_path)
    column_names = ['Battle_Score', 'CatBoost', 'LightGBM', 'MLP', 'SVM', 'XGBoost']

    if not all(col in df.columns for col in column_names):
        print("Required columns not found in the file.")
        return

    df['final_pred_score'] = sum(df[col] * weight for col, weight in zip(column_names, weights))
    df['final_pred'] = df['final_pred_score'].apply(lambda x: 1 if x >= threshold_val else 0)

    cols = list(df.columns[df.columns != 'True Label']) + ['True Label']
    df = df[cols]

    output_dir = os.path.join('eval', platform)
    os.makedirs(output_dir, exist_ok=True)

    base_name, _ = os.path.splitext(os.path.basename(adaptlink_file))
    output_file = os.path.join(output_dir, f"{base_name}_pred.csv")

    df.to_csv(output_file, index=False)
    print(f"Prediction file saved at {output_file}")

    df['TP'] = ((df['final_pred'] == 1) & (df['True Label'] == 1)).astype(int)
    df['FP'] = ((df['final_pred'] == 1) & (df['True Label'] == 0)).astype(int)
    df['TN'] = ((df['final_pred'] == 0) & (df['True Label'] == 0)).astype(int)
    df['FN'] = ((df['final_pred'] == 0) & (df['True Label'] == 1)).astype(int)

    recall = df['TP'].sum() / (df['TP'].sum() + df['FN'].sum())
    precision = df['TP'].sum() / (df['TP'].sum() + df['FP'].sum())
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (df['TP'].sum() + df['TN'].sum()) / len(df)

    fpr, tpr, _ = roc_curve(df['True Label'], df['final_pred'])
    auc_roc = auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(df['True Label'], df['final_pred']).ravel()
    npv = tn / (tn + fn)
    oei = ((tp + tn + fp + fn) / (tp + fp)) * (recall * npv)

    results_df = pd.DataFrame({
        'Metric': ['TP', 'TN', 'FP', 'FN', 'Accuracy', 'Recall', 'NPV', 'AUC-ROC', 'Operational Efficiency Index'],
        'Value': [tp, tn, fp, fn, accuracy, recall, npv, auc_roc, oei]
    })
    results_df.to_csv(os.path.join(output_dir, f"{base_name}_results.csv"), index=False)

    print(results_df.to_string(index=False))

if __name__ == "__main__":
    mobile_weights = [0.05103119, 0.03980747, 0.52915914, 0.06095147, 0.02168069, 0.29737004]
    mobile_threshold = 0.5256941311073032

    pc_weights = [0.32649401, 0.04408209, 0.05239452, 0.09593209, 0.04060686, 0.44049043]
    pc_threshold = 0.5428112966660973

    if len(sys.argv) == 2:
        platform = sys.argv[1]
        if platform == 'mobile':
            weights = mobile_weights
            threshold = mobile_threshold
        elif platform == 'pc':
            weights = pc_weights
            threshold = pc_threshold
        else:
            print("Platform must be either 'mobile' or 'pc'.")
            sys.exit(1)
    elif len(sys.argv) == 9:
        platform = sys.argv[1]
        weights = [float(sys.argv[i]) for i in range(2, 8)]
        threshold = float(sys.argv[8])
    else:
        print("Usage: python evaluation.py [platform] [weight1 weight2 weight3 weight4 weight5 weight6 threshold]")
        print("Example: python evaluation.py mobile 0.051 0.039 0.529 0.060 0.021 0.297 0.525")
        sys.exit(1)

    process_file(platform, weights, threshold)
