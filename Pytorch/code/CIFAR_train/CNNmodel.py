import torch.nn as nn
import torch

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


#测试网络模型正确性
if __name__=='__main__':
    cifar=cifar10()
    #用一组正确的输入输出来验证
    input=torch.ones((64,3,32,32))
    output=cifar(input)
    print(output.shape)#torch.Size([64, 10]),意味着返回64行*10列，中间的数据是分类概率



