import pandas as pd

# 指定三个CSV文件的路径
csv_file_1 = 'final_train.csv'
csv_file_2 = 'final_test.csv'
csv_file_3 = 'final_val.csv'

# 读取CSV文件
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)
df3 = pd.read_csv(csv_file_3)

# 将三个DataFrame合并成一个
combined_df = pd.concat([df1, df2, df3])

# 将合并后的DataFrame保存为新的CSV文件
combined_df.to_csv('all.csv', index=False)

print('合并完成，新的CSV文件已保存为 all.csv')
