import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1.0, -0.5], 
                    [-2.0, 3.0]])

input = torch.reshape(input, (-1, 1, 2, 2))

#ReLU的作用：非线性激活函数，input<0时输出0，input>0时输出input，增加网络的非线性表达能力
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU() #inplace=False,是否在原地进行操作，默认为False

    def forward(self, input):
        return self.relu(input)

relu = ReLU()
output = relu(input)
print("input:", input)
print("output:", output)