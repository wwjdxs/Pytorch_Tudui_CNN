import torch

output = torch.tensor([[0.1,0.2],
                       [0.05,0.4]])

print(output.argmax(1))#x-axis
print(output.argmax(0))#y-axis

pred=output.argmax(1)

targets=torch.tensor([0,1])
print((pred==targets).sum())