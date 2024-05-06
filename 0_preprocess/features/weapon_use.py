import csv
import os
import numpy as np
import re
from collections import defaultdict

def weaponuse(player_file_path):
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return {}

        player_id = rows[0]['PlayerID']
        weapon_changes = 0
        weapon_diversity = set()
        weapon_fire_counts = defaultdict(int)
        weapon_hit_counts = defaultdict(int)

        last_weapon_id = None
        for row in rows:
            weapon_id = row['WeaponID']
            hit_weapon_id = row['HitWeaponID']

            # Count weapon changes
            if weapon_id and weapon_id != last_weapon_id:
                weapon_changes += 1
                last_weapon_id = weapon_id

            # Count weapon diversity
            if weapon_id:
                weapon_diversity.add(weapon_id)

            # Count weapon fire and hit
            if weapon_id:
                weapon_fire_counts[weapon_id] += 1
            if hit_weapon_id:
                weapon_hit_counts[hit_weapon_id] += 1

        # Calculate weapon precision and variance
        weapon_precision = {}
        for weapon_id in weapon_fire_counts:
            if weapon_id in weapon_hit_counts:
                weapon_precision[weapon_id] = weapon_hit_counts[weapon_id] / weapon_fire_counts[weapon_id]
            else:
                weapon_precision[weapon_id] = 0
        
        # print(weapon_precision)

        weapon_variance = np.var(list(weapon_precision.values()))

        return {
            'PlayerID': player_id,
            'WeaponChange': weapon_changes,
            'WeaponDiversity': len(weapon_diversity),
            'WeaponVariance': weapon_variance
        }

def save_weaponuse(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_weapon_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing WeaponUse: {player_csv}')
                    player_data = weaponuse(player_file_path)
                    if player_data:
                        player_data['MatchID'] = game_folder
                        game_weapon_data.append(player_data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_WeaponUse_Structure.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'WeaponChange', 'WeaponDiversity', 'WeaponVariance']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_weapon_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_weaponuse(data_directory)
