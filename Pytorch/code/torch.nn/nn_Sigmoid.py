import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Sigmoid的作用：非线性激活函数，将输入映射到(0, 1)区间，常用于二分类问题

#ReLU的作用：非线性激活函数，input<0时输出0，input>0时输出input，增加网络的非线性表达能力
class non_linear(nn.Module):
    def __init__(self):
        super(non_linear, self).__init__()
        self.relu = nn.ReLU() #inplace=False,是否在原地进行操作，默认为False
        self.sigmoid = nn.Sigmoid() 

    def forward(self, input):
        return self.sigmoid(input)

writer = SummaryWriter(r'E:\DeepLearning\deep_learning_pytorch2.14\dataLoader')

step = 0
nl = non_linear()
for data in dataloader:
    imgs, targets = data
    output = nl(imgs)
    input_grid = torchvision.utils.make_grid(imgs, nrow=8)
    output_grid = torchvision.utils.make_grid(output, nrow=8)
    writer.add_image("input", input_grid, global_step=step)
    writer.add_image("output", output_grid, global_step=step)
    step += 1

writer.close()