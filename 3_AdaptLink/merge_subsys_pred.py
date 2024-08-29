import os
import sys
import pandas as pd

def process_files(platform, mode):
    mode_dir = os.path.join(platform, mode)
    
    match_file = f'MatchScope_{mode}.csv'
    battle_file = f'BattleScan_{mode}.csv'
    
    match_path = os.path.join(mode_dir, match_file)
    battle_path = os.path.join(mode_dir, battle_file)
    
    if not os.path.exists(match_path) or not os.path.exists(battle_path):
        print(f"Files {match_file} or {battle_file} do not exist in {mode_dir}.")
        return
    
    df1 = pd.read_csv(match_path)
    df2 = pd.read_csv(battle_path)
    
    df1['ID'] = df1['ID'].astype(str)
    df2['ID'] = df2['ID'].astype(str)
    
    common_ids = pd.merge(df1[['ID']], df2[['ID']], on='ID', how='inner')
    
    df1_reduced = df1.drop(columns=['final_pred'])
    
    df1_reduced.rename(columns=lambda x: x.split('_')[0], inplace=True)
    
    merged_df = pd.merge(common_ids, df1_reduced, on='ID')
    merged_df = pd.merge(merged_df, df2[['ID', 'Pred_Score']], on='ID')
    
    merged_df.rename(columns={'Pred_Score': 'Battle_Score'}, inplace=True)
    
    columns_sorted = sorted([col for col in merged_df.columns if col not in ['ID', 'True Label']])
    columns_final = ['ID'] + columns_sorted + ['True Label']
    merged_df = merged_df[columns_final]

    columns_sorted = sorted([col for col in merged_df.columns if col not in ['ID', 'True Label']])
    ordered_columns = ['ID'] + columns_sorted + ['True Label']
    merged_df = merged_df[ordered_columns]
    
    output_file = f'AdaptLink_{mode}.csv'
    output_path = os.path.join(platform, mode, output_file)
    
    merged_df.to_csv(output_path, index=False)
    print(f"Merged file saved as {output_path}")

def main():
    if len(sys.argv) == 1:
        platforms = ['mobile', 'pc']
    elif len(sys.argv) == 2:
        platforms = [sys.argv[1]]
    else:
        print("Usage: python merge_pred.py [platform]")
        sys.exit(1)
    
    for platform in platforms:
        process_files(platform, 'test')
        process_files(platform, 'val')

if __name__ == "__main__":
    main()
