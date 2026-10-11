import torch
import torch.nn as nn
import torch.nn.functional as F


#二维tensor数据
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

#设定卷积核
kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])


#F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
print(input.shape)
print(kernel.shape)
#torch.Size([5, 5])
#torch.Size([3, 3])


#shape不符合->变换shape
input=torch.reshape(input,[1, 1, 5, 5])
kernel=torch.reshape(kernel,[1, 1, 3, 3])
print(input.shape)
print(kernel.shape)
'''
input  [1, 1, 5, 5]
         │  │  │  │
         │  │  └──┴─ 高和宽
         │  └─────── 1个输入通道
         └────────── 1张图片

kernel [1, 1, 3, 3]
         │  │  │  │
         │  │  └──┴─ 3×3卷积核
         │  └─────── 接收1个输入通道in_channels
         └────────── 产生1个输出通道out_channels=kernel数量,最终合并运算结果

'''

#卷积函数
output=F.conv2d(input,kernel,stride=1)#每次移动一格
print(output)

output2=F.conv2d(input,kernel,stride=2)#每次移动一格
print(output2)

#padding
output3=F.conv2d(input,kernel,stride=1,padding=1)#5*5 -> 7*7,默认填充0
print(output3)