import csv
import os
import re

def abilityused(player_file_path):
    battle_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return battle_data

        player_id = rows[0]['PlayerID']
        battle_num = 0
        ability_counts = {}
        t_start = t_end = None

        for i, row in enumerate(rows):
            if row['HitRate']:
                battle_num += 1

                if not ability_counts:
                    ability_counts['0'] = 0
                
                total_used_number = sum(ability_counts.values())
                ability_ids = '_'.join(ability_counts.keys())
                t_battle = t_end - t_start if t_start and t_end else 0
                ability_used = total_used_number / t_battle if t_battle else 0
                battle_data.append({
                    'PlayerID': player_id,
                    'BattleNum': battle_num,
                    'usednumber': total_used_number,
                    't_battle': t_battle,
                    'AbilityID': ability_ids,
                    'AbilityUsed': ability_used
                })

                ability_counts = {}
                t_start = t_end = None
                continue

            if 'ServerTime' in row and row['ServerTime'] and t_start is None:
                t_start = float(row['ServerTime'])

            if 'ServerTime' in row and row['ServerTime']:
                t_end = float(row['ServerTime'])

            if 'SkillID' in row and row['SkillID']:
                ability_id = row['SkillID']
                if ability_id not in ability_counts:
                    ability_counts[ability_id] = 0
                ability_counts[ability_id] += 1

    return battle_data

def save_abilityused(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_ability_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing AbilityUsed: {player_csv}')
                    player_battles_data = abilityused(player_file_path)
                    for data in player_battles_data:
                        data['MatchID'] = game_folder
                        game_ability_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_AbilityUsed.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'usednumber', 't_battle', 'AbilityID', 'AbilityUsed']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_ability_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_abilityused(data_directory)