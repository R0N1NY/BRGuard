import csv
import os
import re
from collections import defaultdict

import csv
from collections import defaultdict

def calculate_precision_for_player(player_file_path):
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        shooting_rounds = []
        current_round = {}
        player_id = None

        for row in reader:
            if not player_id:
                player_id = row['PlayerID']
            weapon_id = row['WeaponID']
            weapon_ammo = int(row['WeaponCurrentAmmo']) if row['WeaponCurrentAmmo'] else None
            server_time = float(row['ServerTime'])
            damage_value = row['DamageValue']
            current_weapon_id = row['CurrentWeaponId']

            # Identify the start and end of a shooting round
            if weapon_id and (not current_round or weapon_id != current_round.get('WeaponID') 
                              or (current_round and weapon_ammo > current_round['min_ammo'])):
                if current_round:
                    # Finalize shots count for the previous round
                    current_round['shots'] = current_round['start_ammo'] - current_round['min_ammo']
                    shooting_rounds.append(current_round)
                current_round = {
                    'WeaponID': weapon_id,
                    'start_ammo': weapon_ammo,
                    'min_ammo': weapon_ammo,
                    'start_time': server_time,
                    'end_time': server_time,
                    'shots': 0,
                    'hits': 0,
                    'hit_times': set()
                }
            elif weapon_id and weapon_ammo is not None:
                current_round['min_ammo'] = min(current_round['min_ammo'], weapon_ammo)
                current_round['end_time'] = server_time

            # Check for hits
            if damage_value and current_weapon_id == current_round.get('WeaponID'):
                if current_round['start_time'] <= server_time <= current_round['end_time'] + 2:
                    current_round['hits'] += 1
                    current_round['hit_times'].add(server_time)

        if current_round:
            current_round['shots'] = current_round['start_ammo'] - current_round['min_ammo']
            shooting_rounds.append(current_round)

        total_shots, total_hits = 0, 0
        for round in shooting_rounds:
            total_shots += round['shots']
            total_hits += len(round['hit_times'])

        precision = total_hits / total_shots if total_shots > 0 else 0
        return {
            'PlayerID': player_id,
            'shots': total_shots,
            'hits': total_hits,
            'precision': precision
        }

def calculate_precision(data_dir):
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
                    player_precision_data = calculate_precision_for_player(player_file_path)
                    player_precision_data['MatchID'] = game_folder  # Add MatchID to player's data
                    game_precision_data.append(player_precision_data)

            # Output the aggregated data to a CSV file
            output_file_path = os.path.join(game_folder_path, f'{game_folder}_precision.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'shots', 'hits', 'precision']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_precision_data)

data_directory = '/home/chenxin/Farlight-84/dataset/test'
calculate_precision(data_directory)
