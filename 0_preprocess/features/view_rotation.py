import csv
import os
import re
import numpy as np

def calc_rotation_diff(yaw1, yaw2):
    diff = yaw2 - yaw1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return abs(diff)

def view(player_file_path):
    view_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return view_data

        player_id = rows[0]['PlayerID']
        battle_num = 0
        rotation_data = []
        times = []
        t_start = t_end = None

        for i, row in enumerate(rows):
            if row['HitRate']:
                battle_num += 1

                if rotation_data:
                    rotation_diffs = [calc_rotation_diff(rotation_data[j], rotation_data[j + 1]) for j in range(len(rotation_data) - 1)]
                    rotation_speeds = np.array(rotation_diffs) / np.diff(times)
                    rotation_range = max(rotation_diffs)
                    rotation_changes = np.sum(np.abs(np.diff(rotation_speeds)) > 0)
                    # print(rotation_changes)
                    # rotation_frequency = rotation_changes / (times[-1] - times[0]) if times[-1] - times[0] > 0 else 0
                    # print(times[-1] - times[0])
                    rotation_stability = np.std(rotation_speeds) if len(rotation_speeds) > 0 else 0

                    view_data.append({
                        'PlayerID': player_id,
                        'BattleNum': battle_num,
                        'RotationSpeed': np.mean(rotation_speeds) if len(rotation_speeds) > 0 else 0,
                        'RotationMaxRange': rotation_range,
                        # 'rotationFrequency': rotation_frequency,
                        'RotationStability': rotation_stability
                    })

                rotation_data = []
                times = []
                t_start = t_end = None
                continue

            if 'ServerTime' in row and row['ServerTime']:
                t = float(row['ServerTime'])
                times.append(t)

                if t_start is None:
                    t_start = t
                t_end = t

            if 'Rotation.Yaw' in row and row['Rotation.Yaw']:
                rotation_data.append(float(row['Rotation.Yaw']))

    return view_data

def save_view(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_view_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing ViewRotation: {player_csv}')
                    player_view_data = view(player_file_path)
                    for data in player_view_data:
                        data['MatchID'] = game_folder
                        game_view_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_ViewRotation.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'RotationSpeed', 'RotationMaxRange', 'RotationStability']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_view_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_view(data_directory)
