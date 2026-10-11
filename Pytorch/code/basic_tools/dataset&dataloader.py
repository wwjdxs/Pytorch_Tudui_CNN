#pyright: reportMissingImports=false

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class Mydata(Dataset):
    def __init__(self, root_dir, label_dir): #初始全局变量self及其属性
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path=os.listdir(self.path)

    def __len__(self): #显示data长度
        return len(self.img_path)

    def __getitem__(self, idx): #根据索引返回数据
        img_name = self.img_path[idx]
        img_item_path=os.path.join(self.path,img_name)#root+lable+name
        img=Image.open(img_item_path) 
        label=self.label_dir
        return img, label


ants_dataset=Mydata(root_dir='data/train',label_dir='ants')

print(ants_dataset.__getitem__(0)) #测试数据集（self不用填写）
print(ants_dataset[0]) #两种写法等价


print(ants_dataset.__len__()) #测试数据集长度
print(len(ants_dataset)) #两种写法等价