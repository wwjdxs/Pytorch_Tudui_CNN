from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

import torch
import torchvision
import torch.nn as nn


vgg16 = torchvision.models.vgg16(weights=None)#参数没有经过预训练的vgg16模型

#保存方式1：保存完整模型对象
torch.save(vgg16, MODEL_DIR / 'vgg16_model1.pth')


#保存方式2：
torch.save(vgg16.state_dict(), MODEL_DIR / 'vgg16_model2.pth')#只保存模型参数


#陷阱

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = MyModel()
torch.save(model, MODEL_DIR / 'mymodel.pth')  #保存完整模型对象



