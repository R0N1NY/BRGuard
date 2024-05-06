import csv
import os
import re

def status(player_file_path):
    battle_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return battle_data

        player_id = rows[0]['PlayerID']
        battle_num = 1
        alert_count = 0
        expose_count = 0
        row_count = 0

        for row in rows:
            if row['bDead'] == '1':
                continue

            if row['HitRate']:
                if row_count > 0:
                    alert_percent = (alert_count / row_count)
                    expose_percent = (expose_count / row_count)
                else:
                    alert_percent = 0
                    expose_percent = 0
                # print(alert_count, expose_count, row_count)
                battle_data.append({
                    'PlayerID': player_id,
                    'BattleNum': battle_num,
                    'AlertPercent': alert_percent,
                    'ExposePercent': expose_percent
                })

                battle_num += 1
                alert_count = 0
                expose_count = 0
                row_count = 0
            else:
                row_count += 1
                if row['bIsWalk'] == '1':
                    alert_count += 1
                if row['bIsJump'] == '1' or row['bIsFalling'] == '1' or row['CurrentVehicleName']:
                    expose_count += 1

    return battle_data

def save_status(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_status_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing StatusPercent: {player_csv}')
                    player_battles_data = status(player_file_path)
                    for data in player_battles_data:
                        data['MatchID'] = game_folder
                        game_status_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_StatusPercent.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'AlertPercent', 'ExposePercent']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_status_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_status(data_directory)
