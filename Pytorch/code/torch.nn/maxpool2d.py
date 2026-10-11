import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#最大池化的作用：保留数据特征但是减少数据量，降低计算量，防止过拟合
#卷积-池化-非线性激活

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

class MaxPool2d(nn.Module):
    def __init__(self):
        super(MaxPool2d, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, ceil_mode=False)  # kernel_size=3, stride=3, padding=0, dilation=1, return_indices=False, ceil_mode=False
        

    def forward(self, input):
        return self.maxpool2d(input)

maxpool2d = MaxPool2d()

writer = SummaryWriter(r'E:\DeepLearning\deep_learning_pytorch2.14\dataLoader')

step = 0
for data in dataloader:
    imgs, targets = data
    output = maxpool2d(imgs)#channel数不变，宽高缩小
    input_grid = torchvision.utils.make_grid(imgs, nrow=8)
    output_grid = torchvision.utils.make_grid(output, nrow=8)
    writer.add_image("input", input_grid, global_step=step)
    writer.add_image("output", output_grid, global_step=step)
    step += 1

writer.close()