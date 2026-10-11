#dataloader 从 dataset 中加载数据

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
#导入 dataset

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)#取64张图片作为一个 batch打包，shuffle=False 表示不打乱数据

#测试数据集第一张图片及label
img,label=test_dataset[0]#取出第一张图片和标签
print(img.shape)#打印图片的形状
print(label)#打印标签


# 取出test_loader中的第一批数据
writer=SummaryWriter('dataLoader')
# step=0
# for data in test_loader:
#     imgs,labels=data
#     #print(labels)
#     #imgs.shape=torch.Size([4, 3, 32, 32])
#     writer.add_images("test_data",imgs,step)
#     step+=1


#shuffle的打乱作用
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,labels=data
        writer.add_images('epoch:{}'.format(epoch),imgs,step)
        step+=1

writer.close()