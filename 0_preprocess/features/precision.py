import csv
import os
import re

def precision(player_file_path):
    battle_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        first_row = next(reader, None)
        if not first_row:
            return battle_data

        player_id = first_row['PlayerID']
        battle_num = 1
        for row in reader:
            if row['HitRate']:
                player_data = {
                    'PlayerID': player_id,
                    'shots': row['WeaponShootNun'],
                    'hits': row['ShootHitNum'],
                    'Precision': row['HitRate'],
                    'BattleNum': battle_num
                }
                battle_data.append(player_data)
                battle_num += 1
    return battle_data


def save_precision(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_precision_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing Precision: {player_csv}')
                    player_battles_data = precision(player_file_path)
                    for data in player_battles_data:
                        data['MatchID'] = game_folder
                        game_precision_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_Precision.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'shots', 'hits', 'Precision']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_precision_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_precision(data_directory)
