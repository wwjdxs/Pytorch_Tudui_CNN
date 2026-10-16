from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

import torch
import torchvision
import torch.nn as nn


#导入方式1->保存方式1
# 完整模型文件仅在来源可信时使用 weights_only=False
model = torch.load(MODEL_DIR / 'vgg16_model1.pth', weights_only=False, map_location="cpu")
#print(model)  #打印模型结构

vgg16 = torchvision.models.vgg16(weights=None)  #参数没有经过预训练的vgg16模型
vgg16.load_state_dict(torch.load(MODEL_DIR / 'vgg16_model2.pth', weights_only=True, map_location="cpu"))  #导入方式2->保存方式2
#print(vgg16)  #打印模型结构


#陷阱:必须写class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x

#model = MyModel()这句话不用写

model = torch.load(MODEL_DIR / 'mymodel.pth', weights_only=False, map_location="cpu")  #导入方式1->保存方式1
print(model)  #打印模型结构