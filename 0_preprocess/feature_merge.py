import pandas as pd
import os
import sys
import re

if len(sys.argv) != 2:
    print("Usage: python feature_merge.py DATA_DIR")
    sys.exit(1)

data_dir = sys.argv[1]

def merge_battle(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            merged_data = pd.DataFrame()
            for file_name in os.listdir(game_folder_path):
                if file_name.startswith(game_folder) and not file_name.endswith('_Structure.csv'):
                    file_path = os.path.join(game_folder_path, file_name)
                    df = pd.read_csv(file_path)
                    specific_columns = ['t_battle', 'AbilityUsed'] if file_name.endswith('_AbilityUsed.csv') else ['AlertPercent', 'ExposePercent'] if file_name.endswith('_StatusPercent.csv') else ['RotationSpeed', 'RotationMaxRange', 'RotationStability'] if file_name.endswith('_ViewRotation.csv') else [df.columns[-1]]
                    df = df[['PlayerID', 'MatchID', 'BattleNum'] + specific_columns]
                    merged_data = pd.merge(merged_data, df, on=['PlayerID', 'MatchID', 'BattleNum'], how='outer') if not merged_data.empty else df
            merged_data.sort_values(by=['PlayerID', 'MatchID', 'BattleNum'], inplace=True)
            merged_data_columns = ['PlayerID', 'MatchID', 'BattleNum'] + sorted([col for col in merged_data.columns if col not in ['PlayerID', 'MatchID', 'BattleNum']])
            merged_data = merged_data[merged_data_columns]
            merged_data.to_csv(os.path.join(game_folder_path, f'A_{game_folder}_battle.csv'), index=False)
            print(f'Merge battles done: {game_folder}')

def merge_structures(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            battle_df = pd.read_csv(os.path.join(game_folder_path, f'A_{game_folder}_battle.csv'))
            avg_data = battle_df.groupby(['PlayerID', 'MatchID'], as_index=False).mean()
            max_battle_num = battle_df.groupby(['PlayerID', 'MatchID'], as_index=False)['BattleNum'].max()
            avg_data['BattleNum'] = max_battle_num['BattleNum']
            
            structure_files = [f for f in os.listdir(game_folder_path) if f.endswith('_Structure.csv')]
            for file_name in structure_files:
                file_path = os.path.join(game_folder_path, file_name)
                structure_df = pd.read_csv(file_path)
                avg_data = pd.merge(avg_data, structure_df, on=['PlayerID', 'MatchID'], how='left')

            settle_files = [f for f in os.listdir(game_folder_path) if f.endswith('_settle.csv')]
            for file_name in settle_files:
                file_path = os.path.join(game_folder_path, file_name)
                settle_df = pd.read_csv(file_path)
                columns_to_merge = ['PlayerID'] + [col for col in settle_df.columns if col not in ['PlayerID', 'HeroName', 'TeamID']]
                avg_data = pd.merge(avg_data, settle_df[columns_to_merge], on='PlayerID', how='left')
            
            csv_file_pattern = re.compile(r'\d+\.csv$')
            pattern_files = [f for f in os.listdir(game_folder_path) if csv_file_pattern.search(f)]
            player_id_platform_map = {}
            for file_name in pattern_files:
                file_path = os.path.join(game_folder_path, file_name)
                pattern_df = pd.read_csv(file_path, nrows=1, low_memory=False, dtype={25: str, 49: str, 50: str})
                if 'PlayerID' in pattern_df.columns and 'PlatformType' in pattern_df.columns:
                    player_id = pattern_df['PlayerID'].iloc[0]
                    platform_type = pattern_df['PlatformType'].iloc[0]
                    if player_id not in player_id_platform_map:
                        player_id_platform_map[player_id] = platform_type
            
            for player_id, platform_type in player_id_platform_map.items():
                avg_data.loc[avg_data['PlayerID'] == player_id, 'PlatformType'] = platform_type
            
            duplicate_columns = avg_data.columns[avg_data.columns.duplicated(keep=False)]
            if not duplicate_columns.empty:
                avg_data = avg_data.loc[:, ~avg_data.columns.duplicated(keep='first')]
                print(f"Removed duplicate columns: {duplicate_columns.tolist()}")

            fixed_columns = ['PlayerID', 'MatchID']
            other_columns = sorted([col for col in avg_data.columns if col not in fixed_columns])
            avg_data = avg_data[fixed_columns + other_columns]
            
            avg_data.to_csv(os.path.join(game_folder_path, f'A_{game_folder}_match.csv'), index=False)
            print(f'Merge structures done: {game_folder}')

def merge_to_one(data_dir, output_name):
    all_data = pd.DataFrame()
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            file_path = os.path.join(game_folder_path, f'A_{game_folder}_{output_name}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.to_csv(f'{data_dir}_{output_name}Behavior.csv', index=False)

def merge_final_outputs(data_dir):
    print(f'Merging: {data_dir}')
    merge_to_one(data_dir, 'battle')
    merge_to_one(data_dir, 'match')


merge_battle(data_dir)
merge_structures(data_dir)
merge_final_outputs(data_dir)
