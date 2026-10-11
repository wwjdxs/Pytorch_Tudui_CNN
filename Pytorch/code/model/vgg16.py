import torchvision
import torch
import torch.nn as nn   

#train_data=torchvision.datasets.ImageNet(root=r'E:\DeepLearning\deep_learning_pytorch2.14\data', split='train', download=True, transform=torchvision.transforms.ToTensor())
#这个数据集的压缩包146GB


vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16_false = torchvision.models.vgg16(weights=None)

#在torchvision 0.13版本之后，'pretrained’参数被弃用，建议使用’weights’参数代替。'weights’参数接受一个权重枚举值或者None。
#如果传入的参数既不是权重枚举值也不是None，那么将被视为None。
#vgg16_false = torchvision.models.vgg16(weights=None)

#对现有vgg16模型进行微调
#vgg16_true.classifier[6] = nn.Linear(in_features=4096, out_features=10)  #修改最后一层全连接层的输出为10类 

vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=4096, out_features=10))  #添加一个新的全连接层，输出为10类
print(vgg16_true)  #打印模型结构