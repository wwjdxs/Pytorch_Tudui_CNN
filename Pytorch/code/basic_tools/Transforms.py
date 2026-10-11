from os import write

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from PIL import Image
#transforms.transforms,ctrl按住选择后面一个查看help

#python的用法-->tensor数据类型
#通过transforms.ToTensor()去看2个问题
#1.transforms.ToTensor如何使用
#2.Tensor数据类型的优点

img_path=r'E:\DeepLearning\deep_learning_pytorch2.14\data\train\ants\0013035.jpg'
img=Image.open(img_path)
#img是PIL图像数据类型

image_totensor=transforms.ToTensor()#transforms.ToTensor()是一个类，实例化后才能使用
image1=image_totensor(img)#传入PIL或numpy图像数据类型，返回Tensor数据类型

print('img type:',type(img)) #<class 'PIL.JpegImagePlugin.JpegImageFile'>
print('image1 type:',type(image1)) #<class 'torch.Tensor'>

print(dir(image1)) #查看tensor图片数据的属性和方法
#神经网络需要参数

writer=SummaryWriter('log')#class
writer.add_image('img',image1) #将tensor数据类型的图片写入到tensorboard中
