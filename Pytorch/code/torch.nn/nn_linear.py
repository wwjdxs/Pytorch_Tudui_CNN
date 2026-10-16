from pathlib import Path

# 相对于仓库Pytorch目录定位数据和输出，不依赖E盘或启动位置。
PROJECT_DIR = Path(__file__).resolve().parents[2]

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset=torchvision.datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear=nn.Linear(32*32*3, 10)  # 每张图独立输入3072个特征，不把batch大小写进模型

    def forward(self, input):
        return self.linear(input)
        

lin=Linear()
writer=SummaryWriter(str(PROJECT_DIR / 'dataLoader'))

step=0
for data in dataloader:
    imgs, targets = data
    #imgs.shape: batch_size, channel, height, width
    #input=torch.reshape(imgs,(1,1,1,-1))

    output=torch.flatten(imgs, start_dim=1)  # [B,3,32,32] -> [B,3072]，保留batch维度

    output=lin(output)
    
    input_grid = torchvision.utils.make_grid(imgs, nrow=8)
    writer.add_image("input", input_grid, global_step=step)
    writer.add_histogram("output_logits", output, global_step=step)

    step += 1

# 展平后输入：[B,3072]
# 输出：[B,10]；分类得分使用直方图而不是图像显示。

writer.close()