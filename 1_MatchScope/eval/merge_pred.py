import os
import pandas as pd
import sys

pt = sys.argv[1]

def process_files(mode):
    models = ['LogisticRegression', 'MLP', 'RandomForest', 'SVM', 'CatBoost', 'LightGBM', 'XGBoost']

    if mode == 'test':
        parent_dir = f'../{pt}/test'
        output_dir = f'{pt}/test'
        file_suffix = 'test_pred_'
        output_file_suffix = 'test_pred.csv'
    else:
        parent_dir = f'../{pt}/val'
        output_dir = f'{pt}/val'
        file_suffix = 'pred_'
        output_file_suffix = 'val_pred.csv'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model in models:
        files = [f for f in os.listdir(parent_dir) if f.startswith(f'{model}_{file_suffix}') and f.endswith('.csv')]

        dfs = []
        for file in files:
            file_path = os.path.join(parent_dir, file)
            df = pd.read_csv(file_path)
            df = df.rename(columns={'Predicted': file.replace('.csv', '_pred')})
            dfs.append(df)

        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=['ID', 'True Label'])

            prediction_cols = [col for col in merged_df.columns if col.endswith('_pred')]
            merged_df['Predicted'] = merged_df[prediction_cols].mode(axis=1)[0].astype(int)

            final_df = merged_df[['ID', 'Predicted', 'True Label']]
            final_df.to_csv(f'{output_dir}/{model}_{output_file_suffix}', index=False)

def process_final_stats(mode):
    os.chdir(pt)
    os.chdir(mode)

    files = [f for f in os.listdir() if f.endswith('_pred.csv')]

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df = df.rename(columns={'Predicted': file.replace('_pred.csv', '_pred')})
        dfs.append(df)

    final_df = pd.concat(dfs, axis=1)
    final_df = final_df.loc[:,~final_df.columns.duplicated()]

    cols = list(final_df.columns)
    cols.insert(len(cols), cols.pop(cols.index('True Label')))
    final_df = final_df.loc[:, cols]

    prediction_cols = [col for col in final_df.columns if col.endswith('_pred')]
    final_df['final_pred'] = final_df[prediction_cols].mode(axis=1)[0]

    cols = list(final_df.columns)
    cols.insert(cols.index('True Label'), cols.pop(cols.index('final_pred')))
    final_df = final_df[cols]

    final_df.to_csv(f'MatchScope_{mode}.csv', index=False)
    os.chdir('../..')

if len(sys.argv) == 2:
    process_files('test')
    process_final_stats('test')
    process_files('val')
    process_final_stats('val')
elif len(sys.argv) == 3 and sys.argv[2] in ['test', 'val']:
    process_files(sys.argv[2])
    process_final_stats(sys.argv[2])
else:
    print("Usage: python merge_pred.py PlatformType [test/val]")
    sys.exit(1)
