import torch
import torch as np
'''x = np.arange(4)
print(x.shape)
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)#矩阵的转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B==B.T)#对称矩阵转置相等
c = torch.arange(24).reshape(2, 3, 4)#2维度，3行四列
print(c)
d = c.clone()
print(d)d
print(c*d)
A = np.arange(20*2,dtype=torch.float32).reshape(2, 5, 4)
#print(A.shape)
#print(A.sum())
#print(A)
print(A.sum(axis=0))#按照列相加中间维度求和2#axis=0：表示按行方向求和（即对每一列的元素求和）
print(A.sum(axis=1))
print(A.sum(axis=1, keepdims=True))#按照5求和加ture之后不会丢失维度
f = A.sum() / A.numel()
#print(f)
print(A)
print(A/A.sum(axis=1, keepdims=True))#对应元素除以对应元素
x = np.arange(8).reshape(2, 2, 2)
print(x)
print(x.cumsum(axis=2))#[[0+4, 1+5],[2+6, 3+7]] = [[4, 6],[8, 10]]列累加是一列都有+，加的行值，行累加是一行都有+，加的列值
A = torch.arange(20,dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))#按元素乘再求和，标量
print(A)
print(torch.mv(A, x))#矩阵向量积Ax是一个长度为m的列向量，其ith元素是点积atix
B = torch.ones(4, 3, dtype=torch.float32)
print(B)
print(torch.mm(A, B))#矩阵乘法，行乘列加一起'''
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))#根号下9+16，L2范数
print(torch.abs(u).sum())#L1范数
print(torch.norm(torch.ones((4, 9))))

