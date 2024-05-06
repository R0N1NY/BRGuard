import pandas as pd

# 加载CSV文件
csv1 = pd.read_csv('final_test.csv')
csv2 = pd.read_csv('/home/chenxin/Farlight-84/Temporal/dataset/final_test.csv')

# 从两个CSV中提取ID列，并转换为集合
set1 = set(csv1['ID'])
set2 = set(csv2['ID'])

# 找出两个集合的交集
common_ids = set1.intersection(set2)

# 计算共有ID的个数
common_count = len(common_ids)

print(f'共有的不重复ID的个数是：{common_count}')
