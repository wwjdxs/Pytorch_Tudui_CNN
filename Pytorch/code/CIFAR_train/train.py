from pathlib import Path

# 相对于仓库Pytorch目录定位数据和输出，不依赖E盘或启动位置。
PROJECT_DIR = Path(__file__).resolve().parents[2]

import torch
import os
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#引入神经网络模型
from CNNmodel import *#隐式导入
#显式导入：from CNNmodel import cifar as cifar, cifar10 as cifar10, input as input, nn as nn, output as output, torch as torch


#准备数据集
train_data=torchvision.datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=True, download=False, transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=False, download=False, transform=torchvision.transforms.ToTensor())
#官方预先划分好的测试集


#数据集相关信息
train_data_size=len(train_data)
test_data_size=len(test_data)

print("训练数据集的长度：{}".format(train_data_size))#字符串格式化写法
print("测试数据集的长度：{}".format(test_data_size))

#加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#创建模型
cifar=cifar10()

#创建损失函数
loss_fun=nn.CrossEntropyLoss()#交叉熵

#定义优化器

#learning_rate=0.01
learning_rate=1e-2 #1 x (10)^(-2)=0.01
optimizer=torch.optim.SGD(cifar.parameters(), lr=learning_rate)

#设置一些训练网络的参数

#记录训练次数
total_train_step=0

#记录测试次数
total_test_step=0

#训练的轮数
epoch=10

#添加tensorboard
writer = SummaryWriter(str(PROJECT_DIR / 'dataLoader'))

#每轮模型保存路径
save_path=str(Path(__file__).resolve().parent / 'models_cifar10')
if not os.path.exists(save_path):
    os.makedirs(save_path)


for i in range(epoch):
    print("----------第{}论训练开始-----------".format(i+1))

    #训练开始
    cifar.train()#train()函数对模型的部分层有作用
    for imgs,target in train_dataloader:
        output=cifar(imgs)
        loss=loss_fun(output,target)

    #开始优化
        optimizer.zero_grad()#梯度清零
        loss.backward()
        optimizer.step()

        total_train_step+=1

        if total_train_step%100==0:#减少输出次数
            print("训练次数：{}, loss:{}".format(total_train_step,loss.item()))#loss.item()的作用：把tensor数据类型转化为数字（int/float）
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #每一轮训练后测试模型是否达到要求
    #用测试数据集的正确李来评估模型
    cifar.eval()#和train()一样，对特定层起作用
    total_test_loss=0
    correct_count=0
    total_correct_count=0
    with torch.no_grad():#不记录梯度计算图，不更新参数
        for imgs,target in test_dataloader:
            output=cifar(imgs)#每类原始得分logits，不是概率
            loss=loss_fun(output,target)#部分数据的loss
            total_test_loss+=loss.item() * imgs.size(0)
            #loss可以除以一下size 这样得到的是单个数据的平均loss 准确率同理

            #准确率
            correct_count=(output.argmax(1)==target).sum().item()
            total_correct_count+=correct_count
            

    total_test_loss = total_test_loss / test_data_size
    correct_rate=total_correct_count/test_data_size
    print("第{}轮整体测试集上的正确率:{}".format(i+1,correct_rate))
    print("第{}轮整体测试集上的loss:{}".format(i+1,total_test_loss))
    total_test_step+=1
    writer.add_scalar("correct_rate", correct_rate,total_test_step)
    writer.add_scalar("test_loss",total_test_loss,total_test_step)

    #分类问题常用测试
    #Argmax把ouput从矩阵转化成索引，即找到矩阵中的最大值的索引，记为pred
    #用==判断pred和target是否相等，输出结果为T和F的矩阵，T为1，F为0
    #用sum求和，再让正确个数/总个数，得到正确率


    #保存每一轮的模型，找到最好
    
    

    torch.save(cifar.state_dict(),os.path.join(save_path,"cifar{}.pth".format(i)))
    print('模型已保存')

writer.close()
