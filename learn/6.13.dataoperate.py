import os
import pandas as pd
import torch
import torch as np
# 创建数据目录和文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'data.csv')
with open(data_file, 'w', newline='') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取CSV文件，明确指定NA为缺失值
data = pd.read_csv(data_file, na_values=['NA'])
print("原始数据：")
print(data)

# 分离特征和标签
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 对数值列填充平均值
numeric_inputs = inputs.select_dtypes(include=['number'])
inputs[numeric_inputs.columns] = numeric_inputs.fillna(numeric_inputs.mean())
print("\n处理后的数据：")
print(inputs)
# 对分类列填充众数或其他默认值
'''categorical_inputs = inputs.select_dtypes(exclude=['number'])
inputs[categorical_inputs.columns] = categorical_inputs.fillna('unknown')'''
inputs = pd.get_dummies(inputs, dummy_na=True)#进行不同类别的分割
print(inputs)
# 将布尔列转换为浮点数
inputs = inputs.astype(float)  # 关键修改：将所有列转换为float类型
print("\n转换为浮点数后：")
print(inputs)
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)