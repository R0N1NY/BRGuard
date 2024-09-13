import pandas as pd
import numpy as np
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python gen_data.py black_data_path white_data_path")
    sys.exit(1)

black_data_path = sys.argv[1]
white_data_path = sys.argv[2]

black_data = pd.read_csv(black_data_path)
white_data = pd.read_csv(white_data_path)

black_data['isCheater'] = 1
white_data['isCheater'] = 0

black_data['ID'] = black_data.iloc[:, 0].astype(str) + '_' + black_data.iloc[:, 1].astype(str)
white_data['ID'] = white_data.iloc[:, 0].astype(str) + '_' + white_data.iloc[:, 1].astype(str)

black_data = black_data.drop(black_data.columns[[0, 1]], axis=1)
white_data = white_data.drop(white_data.columns[[0, 1]], axis=1)

black_data = black_data[['ID'] + [col for col in black_data.columns if col != 'ID']]
white_data = white_data[['ID'] + [col for col in white_data.columns if col != 'ID']]

combined_df = pd.concat([black_data, white_data]).reset_index(drop=True)

combined_df.fillna(0, inplace=True)

combined_df['group'] = combined_df.groupby('ID').ngroup()

pc_data = combined_df[combined_df['PlatformType'] == 'steam']
mobile_data = combined_df[combined_df['PlatformType'].isin(['android', 'ios'])]

output_dir = 'dataset'
pc_dir = os.path.join(output_dir, 'pc')
mobile_dir = os.path.join(output_dir, 'mobile')
for directory in [output_dir, pc_dir, mobile_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_data(df, test_frac, val_frac, columns_to_delete, seed=1):
    groups = df['group'].unique()  # Ensure the 'group' column is used before it's dropped
    np.random.seed(seed)
    np.random.shuffle(groups)
    num_test = int(len(groups) * test_frac)
    num_val = int(len(groups) * val_frac)
    test_groups = groups[:num_test]
    val_groups = groups[num_test:num_test + num_val]
    train_groups = groups[num_test + num_val:]
    test_df = df[df['group'].isin(test_groups)]
    val_df = df[df['group'].isin(val_groups)]
    train_df = df[df['group'].isin(train_groups)]
    train_df = train_df.drop(columns=columns_to_delete + ['group'], errors='ignore')
    test_df = test_df.drop(columns=columns_to_delete + ['group'], errors='ignore')
    val_df = val_df.drop(columns=columns_to_delete + ['group'], errors='ignore')
    return train_df, test_df, val_df

columns_to_delete = ['AvgPartUpgrade', 'PlayerUpgrade', 'ShieldUpgrade', 'PlatformType']
pc_train, pc_test, pc_val = split_data(pc_data, test_frac=0.15, val_frac=0.15, columns_to_delete=columns_to_delete)
mobile_train, mobile_test, mobile_val = split_data(mobile_data, test_frac=0.15, val_frac=0.15, columns_to_delete=columns_to_delete)

# Reorder columns
cols_order = ['ID'] + sorted([col for col in pc_train.columns if col not in ['ID', 'isCheater']]) + ['isCheater']

pc_train = pc_train[cols_order]
pc_test = pc_test[cols_order]
pc_val = pc_val[cols_order]
mobile_train = mobile_train[cols_order]
mobile_test = mobile_test[cols_order]
mobile_val = mobile_val[cols_order]

pc_train.to_csv(os.path.join(pc_dir, 'final_train.csv'), index=False)
pc_test.to_csv(os.path.join(pc_dir, 'final_test.csv'), index=False)
pc_val.to_csv(os.path.join(pc_dir, 'final_val.csv'), index=False)
mobile_train.to_csv(os.path.join(mobile_dir, 'final_train.csv'), index=False)
mobile_test.to_csv(os.path.join(mobile_dir, 'final_test.csv'), index=False)
mobile_val.to_csv(os.path.join(mobile_dir, 'final_val.csv'), index=False)

print("Data splitting complete.")
