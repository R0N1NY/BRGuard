import pandas as pd

# 指定CSV文件的路径
csv_file_1 = '/home/chenxin/Farlight-84/Temporal/dataset/final_val.csv'
csv_file_2 = 'all.csv'
output_csv_file = 'final_val.csv'

# 读取第一个CSV文件，并找出所有不重复的ID
df1 = pd.read_csv(csv_file_1)
unique_ids_1 = set(df1['ID'])

# 读取第二个CSV文件
df2 = pd.read_csv(csv_file_2)
unique_ids_2 = set(df2['ID'])

# 筛选第二个CSV中包含第一个CSV不重复ID的行
filtered_df2 = df2[df2['ID'].isin(unique_ids_1)]

# 保存筛选后的数据到新的CSV文件
filtered_df2.to_csv(output_csv_file, index=False)

# 计算第一个CSV中的独有ID数量
only_in_first_csv_count = len(unique_ids_1 - unique_ids_2)

print(f'在第一个CSV中但不在第二个CSV中的ID个数是：{only_in_first_csv_count}')
