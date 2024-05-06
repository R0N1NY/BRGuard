import csv
import os
import re
import math

def travelspeed(player_file_path):
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_location_and_time_from_weapon_hit(rows, index, search_upwards=True):
        start_index, end_index, step = (index, -1, -1) if search_upwards else (index, len(rows), 1)
        for j in range(start_index, end_index, step):
            if rows[j]['HitWeaponID']:
                # print(f"Found HitWeaponID at row {j}")
                for k in range(j, -1, -1):
                    if rows[k]['ServerTime'] and rows[k]['LocationX'] and rows[k]['LocationY']:
                        # print(f"Using data from row {k}")
                        t = float(rows[k]['ServerTime'])
                        x = float(rows[k]['LocationX'])
                        y = float(rows[k]['LocationY'])
                        # print(t,x,y)
                        return t, x, y
                break
        return None, None, None

    travel_data = []
    with open(player_file_path, 'r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not rows:
            return travel_data

        player_id = rows[0]['PlayerID']
        battle_num = 0
        t_end = location_x = location_y = None

        for i, row in enumerate(rows):
            if row['HitRate']:
                battle_num += 1
                t_end, location_x, location_y = find_location_and_time_from_weapon_hit(rows, i)

            elif t_end is not None and 'ServerTime' in row and row['ServerTime']:
                t_start_next, location_nextx, location_nexty = find_location_and_time_from_weapon_hit(rows, i + 1, False)
                if t_start_next:
                    t_run = t_start_next - t_end if t_end else 0
                    run_distance = calculate_distance(location_x, location_y, location_nextx, location_nexty)
                    travel_speed = run_distance / t_run if t_run else 0

                    travel_data.append({
                        'PlayerID': player_id,
                        'BattleNum': battle_num,
                        'run_distance': run_distance,
                        't_run': t_run,
                        'TravelSpeed': travel_speed
                    })

                    t_end = location_x = location_y = None

    return travel_data

def save_travelspeed(data_dir):
    for game_folder in os.listdir(data_dir):
        game_folder_path = os.path.join(data_dir, game_folder)
        if os.path.isdir(game_folder_path):
            game_travel_data = []

            csv_file_pattern = re.compile(r'\d+\.csv$')

            for player_csv in os.listdir(game_folder_path):
                if csv_file_pattern.search(player_csv):
                    player_file_path = os.path.join(game_folder_path, player_csv)
                    player_travel_data = travelspeed(player_file_path)
                    for data in player_travel_data:
                        data['MatchID'] = game_folder
                        game_travel_data.append(data)

            if game_travel_data:
                last_battle_num = game_travel_data[-1]['battleNum']
                last_player_id = game_travel_data[-1]['PlayerID']

                extra_row = {
                    'PlayerID': last_player_id,
                    'MatchID': game_folder,
                    'battleNum': last_battle_num + 1,
                    'run_distance': 0,
                    't_run': 0,
                    'TravelSpeed': 0
                }
                game_travel_data.append(extra_row)

            output_file_path = os.path.join(game_folder_path, f'{game_folder}_TravelSpeed.csv')
            with open(output_file_path, 'w', newline='', encoding='utf_8_sig') as output_file:
                fieldnames = ['PlayerID', 'MatchID', 'BattleNum', 'run_distance', 't_run', 'TravelSpeed']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(game_travel_data)

data_directory = '/home/chenxin/Farlight-84/dataset/test'
save_travelspeed(data_directory)
