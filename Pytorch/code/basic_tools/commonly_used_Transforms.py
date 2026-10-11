from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import shutil

# 每次运行前清空项目根目录下的 logs 文件夹
log_dir = Path(__file__).resolve().parent.parent / "logs"
if log_dir.exists():
    shutil.rmtree(log_dir)

writer = SummaryWriter(log_dir=str(log_dir))

#img初始是PIL.Image.Image类型
img=Image.open(r"E:\DeepLearning\deep_learning_pytorch2.14\data\train\bees\85112639_6e860b0469.jpg")
print(img)

#transforms.ToTensor()
totensor = transforms.ToTensor()
img_tensor = totensor(img)
writer.add_image("ToTensor", img_tensor)#tag,image_tensor

#normalize()正则化标准化：调整数据的中心和尺度
#output[channel] = (input[channel] - mean[channel]) / std[channel]

img_normalized = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])(img_tensor)

'''
等价于
normalized = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
img_normalized = normalized(img_tensor)

嵌套调用
'''

#1.三个数依次对应 R、G、B
#2.在使用transforms.Normalize()之前，必须先使用transforms.ToTensor()将图片转换为tensor格式
#3.均值(mean)和标准差(standard deviation)的取值范围是[0,1]
#4.训练集、验证集和推理图片应采用相同的 mean、std
#5.如果使用预训练模型，mean和std应与预训练模型的训练集一致
writer.add_image("Normalized", img_normalized)


#resize()调整图片的尺寸
print(img.size)
img_resize=transforms.Resize((512, 512))(img)#指定大小
print(img_resize)#输出是PIL.Image.Image类型
writer.add_image("Resize", totensor(img_resize))

#compose()组合多个transform操作-resize-2
img_resize2=transforms.Resize((512, 512))
trans_compose=transforms.Compose([img_resize2, totensor])#拼接了resize和tosensor的功能
img_resize2=trans_compose(img)
print(img_resize2)
writer.add_image("Compose", img_resize2)
writer.close()
