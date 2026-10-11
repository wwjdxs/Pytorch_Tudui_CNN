#测试已经训练好的模型
#1、自己找图片搭建测试数据集

from PIL import Image
import torch
import torch.nn as nn
import torchvision


img_path=r'E:\DeepLearning\deep_learning_pytorch2.14\data\test_cifar\test1.png'
img=Image.open(img_path)
img=img.convert('RGB')#保证3 channels

transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
])

img=transform(img)
#print(img.shape)#torch.Size([3, 32, 32])
#batch_size:
img=torch.reshape(img,[1,3,32,32])#[批次, 通道, 高, 宽]

class cifar10(nn.Module):
    def __init__(self):
        super(cifar10,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),    
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),        
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

model=torch.load(r'E:\DeepLearning\deep_learning_pytorch2.14\models_cifar10\cifar7.pth',
                 weights_only=False)

#开始测试：养成习惯
model.eval()
with torch.no_grad():
    output=model(img)
print(output)
print(output.argmax(1))#模型不太行