import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())
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
writer = SummaryWriter(log_dir=r'E:\DeepLearning\deep_learning_pytorch2.14\dataLoader')


#反向传播-loss function
loss_fn = nn.CrossEntropyLoss()#classification loss function

for data in dataloader:
    imgs, target = data
    output = cifar10_model(imgs)
    loss = loss_fn(output, target)
    loss.backward()#反向传播计算梯度->得到weights的梯度->优化器优化模型

    #开始的权重随机生成的，然后反向传播就是为了找到合适的权重使loss最小=最接近target
    








