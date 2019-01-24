import torch as t
# print(t.__version__)

# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)
print(x)

# 以指定的值初始化矩阵
x = t.Tensor([[1, 2], [3, 4]])
print(x)

# 使用[0,1]均匀分布随机初始化二维数组
x = t.rand(5, 3)
print(x)
# 查看x的形状
print(x.size())
# 查看列的个数，两种形式等价
print(x.size()[1],x.size(1))

y = t.rand(5,3)
print(y)

# 加法的第一种写法
print(x + y)

# 加法的第二种写法
print(t.add(x, y))

# 加法的第三种写法：指定加法结果的输出目标为result
result = t.Tensor(5, 3) # 预先分配空间
t.add(x, y, out=result) # 输入到result

print(result)

print('最初y')
print(y)

print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)

print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)

# 注意，函数名后面带下划线_ 的函数会修改Tensor本身。例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。

# Tensor的选取操作与Numpy类似 数组的第一个位置表示行，第二个位置表示列
print(y[:, 1])
# Tensor还支持很多操作，包括数学运算、线性代数、选择、切片等等，其接口设计与Numpy极为相似。更详细的使用方法，会在第三章系统讲解。
# Tensor和Numpy的数组之间的互操作非常容易且快速。对于Tensor不支持的操作，可以先转为Numpy数组处理，之后再转回Tensor

a = t.ones(5) # 新建一个全1的Tensor
print(a)