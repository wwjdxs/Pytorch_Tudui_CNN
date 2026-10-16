from pathlib import Path

# 相对于仓库Pytorch目录定位数据和输出，不依赖E盘或启动位置。
PROJECT_DIR = Path(__file__).resolve().parents[2]

import torch
from torch import nn
import torchvision

dataset=torchvision.datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=True, download=False, transform=torchvision.transforms.ToTensor())
dataloader=torch.utils.data.DataLoader(dataset, batch_size=1)


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
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


#反向传播-loss function
loss_fn = nn.CrossEntropyLoss()#classification loss function

#设置优化器
optimizer = torch.optim.SGD(cifar10_model.parameters(), lr=0.01)#model.parameters()是模型的参数，lr是学习率

for epoch in range(10):

    running_loss = 0.0 #每一轮学习的loss累加

    for imgs, target in dataloader:
        optimizer.zero_grad()#梯度清零，防止梯度累加
        output = cifar10_model(imgs)
        loss = loss_fn(output, target)
        loss.backward()#反向传播计算梯度->得到weights的梯度->优化器优化模型
        optimizer.step()#根据梯度更新模型参数
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/10], Loss: {running_loss / len(dataloader):.4f}')
#开始的权重随机生成的，然后反向传播就是为了找到合适的权重使loss最小=最接近target

'''

这个要很久，电脑卡顿，建议使用GPU训练，或者减少数据集的大小，或者减少epoch的数量。

'''
    








