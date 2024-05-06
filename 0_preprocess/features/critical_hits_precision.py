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
                    'headshots': row['HeadShootNum'],
                    'hits': row['ShootHitNum'],
                    'CHPrecision': row['HitHeadRate'],
                    'BattleNum': battle_num
                }
                battle_data.append(player_data)
                battle_num += 1
    return battle_data


def save_cprecision(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_precision_data = []

            # Regular expression to match files that end with a number before .csv
            csv_file_pattern = re.compile(r'\d+\.csv$')

            # Iterate through each player's CSV file in the game folder
            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing CHPrecision: {player_csv}')
                    player_battles_data = precision(player_file_path)
                    for data in player_battles_data:
                        data['MatchID'] = game_folder  # Add MatchID to player's data
                        game_precision_data.append(data)

            # Output the aggregated data to a CSV file
            output_file_path = os.path.join(game_folder_path, f'{game_folder}_CHPrecision.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'headshots', 'hits', 'CHPrecision']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_precision_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_cprecision(data_directory)
