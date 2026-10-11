#神经网络的搭建
#nn.Module的使用   nn= neural network

# import torch.nn as nn
# import torch.nn.functional as F

#类的继承：定义自己的神经网络
# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__() 调用父类初始化函数
#         以下自定义
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  #先对x进行一个卷积，再非线性处理
#         return F.relu(self.conv2(x)) #重复上述步骤

#forward：Define the computation performed at every call.
#         Should be overridden by all subclasses.


import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x+=1
        return x


mynn=MyNN()
input=torch.tensor(1)
output=mynn(input)
print(output)