import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear=nn.Linear(64*32*32*3, 10)#in_features=64*32*32*3, out_features=10

    def forward(self, input):
        return self.linear(input)
        

lin=Linear()
writer=SummaryWriter(r"E:\DeepLearning\deep_learning_pytorch2.14\dataLoader")

step=0
for data in dataloader:
    imgs, targets = data
    #imgs.shape: batch_size, channel, height, width
    #input=torch.reshape(imgs,(1,1,1,-1))

    output=torch.flatten(imgs)#展平变成一行:torch.Size([64*32*32*3])

    output=lin(output)
    
    input_grid = torchvision.utils.make_grid(imgs, nrow=8)
    output_grid = torchvision.utils.make_grid(output, nrow=8)
    writer.add_image("input", input_grid, global_step=step)
    writer.add_image("output", output_grid, global_step=step)

    step += 1

#after reshape, input.shape: torch.Size([1, 1, 1, 64*32*32*3])
#output.shape: torch.Size([1, 1, 1, 10])

writer.close()