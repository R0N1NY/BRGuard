import pandas as pd
import numpy as np
import os
import sys

if len(sys.argv) != 4:
    print("Usage: python gen_data.py black_data_path white_data_path MatchScope_dataset_path")
    sys.exit(1)

black_data_path = sys.argv[1]
white_data_path = sys.argv[2]
MatchScope_dataset_path = sys.argv[3]

black_data = pd.read_csv(black_data_path)
white_data = pd.read_csv(white_data_path)

black_data['isCheater'] = 1
white_data['isCheater'] = 0

black_data['ID'] = black_data.iloc[:, 0].astype(str) + '_' + black_data.iloc[:, 1].astype(str)
white_data['ID'] = white_data.iloc[:, 0].astype(str) + '_' + white_data.iloc[:, 1].astype(str)

black_data = black_data.drop(black_data.columns[[0, 1]], axis=1)
white_data = white_data.drop(white_data.columns[[0, 1]], axis=1)

combined_df = pd.concat([black_data, white_data]).reset_index(drop=True)
combined_df.fillna(0, inplace=True)
combined_df['group'] = combined_df.groupby('ID').ngroup()

previous_ids = {}
for folder in ['pc', 'mobile']:
    path = os.path.join(MatchScope_dataset_path, folder)
    for split in ['final_train.csv', 'final_test.csv', 'final_val.csv']:
        file_path = os.path.join(path, split)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            key = (folder, split.split('_')[1].split('.')[0])
            previous_ids[key] = temp_df['ID'].unique()

output_dir = 'dataset'
for key, ids in previous_ids.items():
    folder, split = key
    df_split = combined_df[combined_df['ID'].isin(ids)]
    
    df_split = df_split.drop(columns=['group'])
    
    cols = sorted([col for col in df_split.columns if col not in ['ID', 'isCheater']])
    df_split = df_split[['ID'] + cols + ['isCheater']]

    save_path = os.path.join(output_dir, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_split.to_csv(os.path.join(save_path, f"final_{split}.csv"), index=False)

print("Data splitting complete.")
