import csv
import os
import re

def hitdistance(player_file_path):
    battle_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return battle_data

        player_id = rows[0]['PlayerID']
        battle_num = 1
        total_distance = 0
        hit_count = 0

        for i, row in enumerate(rows):
            if row['HitRate']:
                if hit_count > 0:
                    hit_distance = total_distance / hit_count
                else:
                    hit_distance = 0

                battle_data.append({
                    'PlayerID': player_id,
                    'BattleNum': battle_num,
                    'totaldistance': total_distance,
                    'hits': hit_count,
                    'HitDistance': hit_distance
                })

                battle_num += 1
                total_distance = 0
                hit_count = 0
                continue

            if row['HitWeaponID']:
                # Find the nearest row with DistanceBetweenActors value
                for j in range(i, -1, -1):
                    if rows[j]['DistanceBetweenActors']:
                        distance = float(rows[j]['DistanceBetweenActors'])
                        total_distance += distance
                        hit_count += 1
                        break

    return battle_data

def save_distance(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_distance_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    print(f'Processing HitDistance: {player_csv}')
                    player_battles_data = hitdistance(player_file_path)
                    for data in player_battles_data:
                        data['MatchID'] = game_folder
                        game_distance_data.append(data)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_HitDistance.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'totaldistance', 'hits', 'HitDistance']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_distance_data)

# data_directory = '/home/chenxin/Farlight-84/dataset/test'
# save_distance(data_directory)
