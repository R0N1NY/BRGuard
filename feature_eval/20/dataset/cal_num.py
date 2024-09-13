import pandas as pd

file_path = '/home/chenxin/Farlight84/1_MatchScope/dataset/pc/final_train.csv'
data = pd.read_csv(file_path)

# 统计isCheater为1的unique ID个数
cheaters = data[data['isCheater'] == 1]['ID'].nunique()

# 统计isCheater为0的unique ID个数
non_cheaters = data[data['isCheater'] == 0]['ID'].nunique()

print(f"Unique ID count for cheaters (isCheater = 1): {cheaters}")
print(f"Unique ID count for non-cheaters (isCheater = 0): {non_cheaters}")
