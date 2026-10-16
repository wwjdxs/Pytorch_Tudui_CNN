from pathlib import Path

# 相对于仓库Pytorch目录定位数据和输出，不依赖E盘或启动位置。
PROJECT_DIR = Path(__file__).resolve().parents[2]

from torch.utils.tensorboard import SummaryWriter
import time

# 每次运行创建独立日志目录，保留旧记录。
log_dir = Path(__file__).resolve().parent.parent / "logs" / (Path(__file__).stem + "_" + str(time.time_ns()))

writer = SummaryWriter(log_dir=str(log_dir))

#使用torchvision.datasets中的数据集
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#用transforms对数据集进行批量预处理
#定义一系列的处理方法
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),#调整图片大小
    transforms.ToTensor(),#将PIL.Image.Image类型转换为tensor类型
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])#正则化标准化
])

#使用datasets.CIFAR10()下载CIFAR10数据集，并使用transform参数对数据集进行预处理
trans_set=datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=True, download=True, transform=data_transform)
test_set=datasets.CIFAR10(root=str(PROJECT_DIR / 'data'), train=False, download=True, transform=data_transform)

# #datasets的使用

# print(test_set[0])#输出是一个元组，包含图片和标签
# print(test_set.classes)#输出类别名称列表

# img, label = test_set[0]
# print(img)
# print(label)
# print(test_set.classes[label])#输出类别名称

for i in range(5):
    img, label = test_set[i]
    writer.add_image(f"Image_{i}", img, i)  # tag, image_tensor, global_step
    #writer.add_image("Test_images", img, i) 这样也可以，不过滑动


    writer.add_text(f"Label_{i}", f"Class: {test_set.classes[label]}")

writer.close()
