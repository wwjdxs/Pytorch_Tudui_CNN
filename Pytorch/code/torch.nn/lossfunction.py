import torch
import torch.nn as nn

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

input=torch.reshape(input, (1,1,1,3))#1指保持前面数据不变，3指拓展成3维
target=torch.reshape(target, (1,1,1,3))

print("Input:", input)
print("Target:", target)

loss = nn.MSELoss()
output = loss(input, target)
print("Loss:", output)