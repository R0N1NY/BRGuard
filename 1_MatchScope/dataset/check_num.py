import pandas as pd

# 替换这里的'path_to_your_file.csv'为你的CSV文件路径
file_path = 'pc/final_train.csv'

# 读取CSV文件
df = pd.read_csv(file_path)

# 确保ID和isCheater列的数据类型正确
df['ID'] = df['ID'].astype(str)
df['isCheater'] = df['isCheater'].astype(int)

# 去重，保留ID和isCheater的唯一组合
df_unique = df.drop_duplicates(subset=['ID', 'isCheater'])

# 计算isCheater为1和0的唯一ID数量
cheaters_count = df_unique[df_unique['isCheater'] == 1]['ID'].nunique()
non_cheaters_count = df_unique[df_unique['isCheater'] == 0]['ID'].nunique()

print(f'Unique IDs with isCheater=1: {cheaters_count}')
print(f'Unique IDs with isCheater=0: {non_cheaters_count}')
