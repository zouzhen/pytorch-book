'''
pytorch的基本知识
'''
# import torch as t
# import numpy as np
# # print(t.__version__)

# # 构建 5x3 矩阵，只是分配了空间，未初始化
# x = t.Tensor(5, 3)
# print(x)

# # 以指定的值初始化矩阵
# x = t.Tensor([[1, 2], [3, 4]])
# print(x)

# # 使用[0,1]均匀分布随机初始化二维数组
# x = t.rand(5, 3)
# print(x)
# # 查看x的形状
# print(x.size())
# # 查看列的个数，两种形式等价
# print(x.size()[1],x.size(1))

# y = t.rand(5,3)
# print(y)

# # 加法的第一种写法
# print(x + y)

# # 加法的第二种写法
# print(t.add(x, y))

# # 加法的第三种写法：指定加法结果的输出目标为result
# result = t.Tensor(5, 3) # 预先分配空间
# t.add(x, y, out=result) # 输入到result

# print(result)

# print('最初y')
# print(y)

# print('第一种加法，y的结果')
# y.add(x) # 普通加法，不改变y的内容
# print(y)

# print('第二种加法，y的结果')
# y.add_(x) # inplace 加法，y变了
# print(y)

# # 注意，函数名后面带下划线_ 的函数会修改Tensor本身。例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。

# # Tensor的选取操作与Numpy类似 数组的第一个位置表示行，第二个位置表示列
# print(y[:, 1])
# # Tensor还支持很多操作，包括数学运算、线性代数、选择、切片等等，其接口设计与Numpy极为相似。更详细的使用方法，会在第三章系统讲解。
# # Tensor和Numpy的数组之间的互操作非常容易且快速。对于Tensor不支持的操作，可以先转为Numpy数组处理，之后再转回Tensor


# a = t.ones(5) # 新建一个全1的Tensor
# print(a)

# b = a.numpy() # Tensor -> Numpy
# print(b)

# a = np.ones(5)
# b = t.from_numpy(a) # Numpy->Tensor
# print(a)
# print(b) 


# # Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。
# b.add_(1) # 以`_`结尾的函数会修改自身
# print(a)
# print(b) # Tensor和Numpy共享内存


# # 如果你想获取某一个元素的值，可以使用scalar.item。 直接tensor[idx]得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.
# scalar = b[0]
# print(scalar)
# print(scalar.size()) #0-dim
# print(scalar.item()) # 使用scalar.item()能从中取出python对象的数值)

# tensor = t.tensor([2]) # 注意和scalar的区别
# print(tensor, scalar)
# print(tensor.size(), scalar.size())

# # 只有一个元素的tensor也可以调用`tensor.item()`
# print(tensor.item(), scalar.item())


# # 此外在pytorch中还有一个和np.array 很类似的接口: torch.tensor, 二者的使用十分类似。
# tensor = t.tensor([3,4]) # 新建一个包含 3，4 两个元素的tensor
# scalar = t.tensor(3)

# old_tensor = tensor
# # new_tensor = t.tensor(old_tensor) 该方法在torch1.0中并非最优的，故采用以下方法进行替换
# new_tensor = t.clone(old_tensor).detach()
# new_tensor[0] = 1111
# print(old_tensor, new_tensor)

# # 需要注意的是，t.tensor()总是会进行数据拷贝，新tensor和原来的数据不再共享内存。所以如果你想共享内存的话，
# # 建议使用torch.from_numpy()或者tensor.detach()来新建一个tensor, 二者共享内存


# new_tensor = old_tensor.detach()
# new_tensor[0] = 1111
# print(old_tensor, new_tensor)


# # Tensor可通过.cuda 方法转为GPU的Tensor，从而享受GPU带来的加速运算。
# # 在不支持CUDA的机器下，下一步还是在CPU上运行
# device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# x = x.to(device)
# y = y.to(device)
# z = x+y
# print(z)
# # 此外，还可以使用tensor.cuda() 的方式将tensor拷贝到gpu上，但是这种方式不太推荐。\
# # 此处可能发现GPU运算的速度并未提升太多，这是因为x和y太小且运算也较为简单，而且将数据从内存转移到显存还需要花费额外的开销。GPU的优势需在大规模数据和复杂运算下才能体现出来。

# # 为tensor设置 requires_grad 标识，代表着需要求导数
# # pytorch 会自动调用autograd 记录操作
# x = t.ones(2, 2, requires_grad=True)

# # 上一步等价于
# # x = t.ones(2,2)
# # x.requires_grad = True
# print(x)

# y = x.sum()
# print(y)
# print(y.grad_fn)
# y.backward() # 反向传播,计算梯度
# # y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# # 每个值的梯度都为1
# print(x.grad) 
# # 注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
# y.backward()
# x.grad
# print(x.grad) 
# y.backward()
# x.grad
# print(x.grad) 

# # 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
# print(x.grad.data.zero_())
# y.backward()
# x.grad
# print(x.grad) 

'''
神经网络
'''
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# 只要在nn.Module的子类中定义了forward函数，backward函数就会自动被实现(利用autograd)。在forward 函数中
# 可使用任何tensor支持的函数，还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致。
# 网络的可学习参数通过net.parameters()返回，net.named_parameters可同时返回可学习的参数及名称。

params = list(net.parameters())
# print(params)
print(len(params))
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# forward函数的输入和输出都是Tensor。
input = t.randn(1, 1, 32, 32)
out = net(input)
print(out.size())

net.zero_grad() # 所有参数的梯度清零
out.backward(t.ones(1,10)) # 反向传播

# 需要注意的是，torch.nn只支持mini-batches，不支持一次只输入一个样本，
# 即一次必须是一个batch。但如果只想输入一个样本，则用 input.unsqueeze(0)将batch_size设为１。
# 例如 nn.Conv2d 输入必须是4维的，形如 nSamples×nChannels×Height×Width 。可将nSample设为1，即 1×nChannels×Height×Width 。

'''
损失函数
nn实现了神经网络中大多数的损失函数，例如nn.MSELoss用来计算均方误差，nn.CrossEntropyLoss用来计算交叉熵损失。
'''

output = net(input)
target = t.arange(0, 10).view(1, 10).float() # 需要将tensor的标量类型由长整形转化为浮点数
print(target)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss) # loss是个scalar

# 运行.backward，观察调用之前和调用之后的grad
net.zero_grad() # 把net中所有可学习参数的梯度清零
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad)

'''
优化器
在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：

weight = weight - learning_rate * gradient
手动实现如下：

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)# inplace 减法
torch.optim中实现了深度学习中绝大多数的优化方法，例如RMSProp、Adam、SGD等，更便于使用，因此大多数时候并不需要手动写上述代码。
'''

import torch.optim as optim
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# 在训练过程中
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad() 

# 计算损失
output = net(input)
loss = criterion(output, target)

#反向传播
loss.backward()

#更新参数
optimizer.step()


'''
数据加载与预处理
在深度学习中数据加载及预处理是非常复杂繁琐的，但PyTorch提供了一些可极大简化和加快数据处理流程的工具。
同时，对于常用的数据集，PyTorch也提供了封装好的接口供用户快速调用，这些数据集主要保存在torchvison中。

torchvision实现了常用的图像数据加载功能，例如Imagenet、CIFAR10、MNIST等，以及常用的数据转换操作，
这极大地方便了数据加载，并且代码具有可重用性。
'''

