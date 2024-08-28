import os
import pandas as pd
import sys

pt = sys.argv[1]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_files(mode):
    parent_dir = os.path.join(os.getcwd(), '..', pt)  # 设置为当前工作目录的上一级目录下的 mobile/pc 文件夹
    output_dir = os.path.join(os.getcwd(), pt)  # 输出目录为当前工作目录下的 mobile/pc 文件夹
    create_directory_if_not_exists(output_dir)  # 确保输出目录存在

    if mode == 'test':
        output_file = 'BattleScan_test.csv'
    else:
        output_file = 'BattleScan_val.csv'

    # 收集所有 results_ 开头的文件夹
    results_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith('results_')]

    merged_df = None
    for results_dir in results_dirs:
        file_path = os.path.join(results_dir, f'{mode}_prediction.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df[['ID', 'Prediction', 'Pred_Score', 'True Label']]
            df = df.rename(columns={'Prediction': f'Prediction_{results_dir}', 'Pred_Score': f'Pred_Score_{results_dir}'})
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=['ID', 'True Label'], how='outer')

    if merged_df is not None:
        pred_score_cols = [col for col in merged_df.columns if col.startswith('Pred_Score')]
        merged_df['Avg_Pred_Score'] = merged_df[pred_score_cols].mean(axis=1)
        merged_df['Prediction'] = (merged_df['Avg_Pred_Score'] > 0.5).astype(int)

        final_df = merged_df[['ID', 'Prediction', 'Avg_Pred_Score', 'True Label']]
        final_df = final_df.rename(columns={'Avg_Pred_Score': 'Pred_Score'})
        final_df.to_csv(os.path.join(output_dir, output_file), index=False)
        print(f"{mode.capitalize()} data merged and saved to {os.path.join(output_dir, output_file)}")

def process_final_stats():
    process_files('test')
    process_files('val')

if len(sys.argv) == 2:
    process_final_stats()
elif len(sys.argv) == 3 and sys.argv[2] in ['test', 'val']:
    process_files(sys.argv[2])
else:
    print("Usage: python merge_pred.py PlatformType [test/val]")
    sys.exit(1)
