import os
import re
import csv
from collections import defaultdict

def parse_part_levels(part_level_list):
    return [max(0, int(level)) if level != '-1' else 0 for level in part_level_list.split('|')]

def calculate_average_upgrade_speed(time_diffs, level_diffs):
    speeds = []
    for time_diff, level_diff in zip(time_diffs, level_diffs):
        if time_diff > 0:
            speed = level_diff / time_diff
            speeds.append(speed)
    return sum(speeds) / len(speeds) if speeds else 0

def levelupgrade(player_file_path):
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return {}

        player_id = rows[0]['PlayerID']
        weapon_part_level_changes = defaultdict(lambda: [0, 0])  # weaponID: [last total level, last update time]
        weapon_part_upgrade_speeds = defaultdict(list)  # weaponID: [(server_time, speed)]
        last_player_level = last_shield_level = 0
        last_player_time = last_shield_time = 0
        player_level_changes, player_time_diffs = [], []
        shield_level_changes, shield_time_diffs = [], []

        for i, row in enumerate(rows):
            if not row['ServerTime'] or not row['CurrentWeaponId'] or not row['CurrentWeaponPartLevelList'] or not row['PlayerLevelNum'] or not row['CurrentShieldLevel']:
                continue

            server_time = float(row['ServerTime'])
            current_weapon_id = row['CurrentWeaponId']
            current_part_levels = sum(parse_part_levels(row['CurrentWeaponPartLevelList']))
            current_player_level = max(0, int(row['PlayerLevelNum']))
            current_shield_level = max(0, int(row['CurrentShieldLevel']))

            # Weapon part upgrade logic
            last_total_level, last_update_time = weapon_part_level_changes[current_weapon_id]
            level_change = current_part_levels - last_total_level
            if level_change != 0:
                time_diff = server_time - last_update_time
                if time_diff > 0:
                    speed = level_change / time_diff
                    weapon_part_upgrade_speeds[current_weapon_id].append((server_time, speed))
                    # print(f'Weapon ID {current_weapon_id} upgrade at ServerTime {server_time}: Level Change = {level_change}, Time Diff = {time_diff}, Speed = {speed}')
                weapon_part_level_changes[current_weapon_id] = [current_part_levels, server_time]

            # Player and shield upgrade logic
            if current_player_level != last_player_level:
                player_level_change = current_player_level - last_player_level
                player_time_diff = server_time - last_player_time
                if player_time_diff > 0:
                    player_level_changes.append(player_level_change)
                    player_time_diffs.append(player_time_diff)
                # print(f'Player Upgrade at ServerTime {server_time}: Level Change = {player_level_change}, Time Diff = {player_time_diff}')
                last_player_time = server_time

            if current_shield_level != last_shield_level:
                shield_level_change = current_shield_level - last_shield_level
                shield_time_diff = server_time - last_shield_time
                if shield_time_diff > 0:
                    shield_level_changes.append(shield_level_change)
                    shield_time_diffs.append(shield_time_diff)
                # print(f'Shield Upgrade at ServerTime {server_time}: Level Change = {shield_level_change}, Time Diff = {shield_time_diff}')
                last_shield_time = server_time

            last_player_level, last_shield_level = current_player_level, current_shield_level

        avg_part_upgrade = sum([speed for speeds in weapon_part_upgrade_speeds.values() for _, speed in speeds]) / sum(len(speeds) for speeds in weapon_part_upgrade_speeds.values()) if weapon_part_upgrade_speeds else 0
        avg_player_upgrade = calculate_average_upgrade_speed(player_time_diffs, player_level_changes)
        avg_shield_upgrade = calculate_average_upgrade_speed(shield_time_diffs, shield_level_changes)

        return {
            'PlayerID': player_id,
            'AvgPartUpgrade': avg_part_upgrade,
            'PlayerUpgrade': avg_player_upgrade,
            'ShieldUpgrade': avg_shield_upgrade
        }

def save_levelupgrade(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_level_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing Upgrade: {player_csv}')
                    player_data = levelupgrade(player_file_path)
                    if player_data:
                        player_data['MatchID'] = game_folder
                        game_level_data.append(player_data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_LevelUpgrade_Structure.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'AvgPartUpgrade', 'PlayerUpgrade', 'ShieldUpgrade']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_level_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_levelupgrade(data_directory)
