import csv
import os
import re

def damage(player_file_path):
    battle_damage_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return battle_damage_data

        player_id = rows[0]['PlayerID']
        battle_num = 1
        battle_damage = 0

        for row in rows:
            if row['HitRate']:
                if battle_damage > 0 or True:
                    battle_damage_data.append({
                        'PlayerID': player_id,
                        'BattleNum': battle_num,
                        'BattleDamage': battle_damage
                    })

                battle_num += 1
                battle_damage = 0
                continue

            damage_value = row.get('DamageValue')
            if damage_value:
                battle_damage += float(damage_value)

    return battle_damage_data

def save_damage(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_damage_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing BattleDamage: {player_csv}')
                    player_battles_damage = damage(player_file_path)
                    for data in player_battles_damage:
                        data['MatchID'] = game_folder
                        game_damage_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_BattleDamage.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'BattleDamage']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_damage_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_damage(data_directory)