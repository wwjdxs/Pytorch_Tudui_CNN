from pathlib import Path

# 相对于仓库Pytorch目录定位数据和输出，不依赖E盘或启动位置。
PROJECT_DIR = Path(__file__).resolve().parents[2]

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter



class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(1024, 64)#1024的计算可以在forward里面写到flatten之后，或者在这里写成1024
        # self.linear2 = nn.Linear(64, 10)

        
        self.seq= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),    
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),        
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )


    def forward(self, x):
        x = self.seq(x)
        return x


cifar10_model = CIFAR10()
writer = SummaryWriter(log_dir=str(PROJECT_DIR / 'dataLoader'))

input=torch.ones(64,3,32,32) # batch_size=64, channels=3, height=32, width=32
writer.add_graph(cifar10_model, input)
writer.close()


# 输出的形状应该是 (64, 10)，表示64个样本，每个样本有10个类别的预测结果





