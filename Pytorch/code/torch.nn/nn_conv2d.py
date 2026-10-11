import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#1、导入数据


dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

#2、搭建神经网络CNN模型

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)#卷积操作：https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    def forward(self,x):
        x=self.conv1(x)
        return x    

cnn=CNN()

writer=SummaryWriter(r'E:\DeepLearning\deep_learning_pytorch2.14\dataLoader')

step=0
for data in dataloader:
    imgs, targets=data
    output=cnn(imgs)
    print('input shape:',imgs.shape) #torch.Size([64, 3, 32, 32])
    print('output shape:',output.shape) #torch.Size([64, 6, 30, 30])

    #卷积输出有 6 个通道，但 TensorBoard/PIL 不能把 6 通道数据当作普通图片显示

    #不太推荐但是可以使用下面的方式把 6 通道数据当作普通图片显示

    output=torch.reshape(output, (-1, 3, 30, 30)) #reshape成3通道数据

    writer.add_image('input', imgs, step)
    writer.add_image('output', output, step)

    step+=1

writer.close()