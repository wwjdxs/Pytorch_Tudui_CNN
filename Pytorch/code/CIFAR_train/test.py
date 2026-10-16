"""旧教程基线的单图测试，不用于加载新版BatchNorm模型。"""
from pathlib import Path
from PIL import Image
import torch
import torchvision
from CNNmodel import cifar10, load_checkpoint_state

PROJECT_DIR = Path(__file__).resolve().parents[2]
img_path = PROJECT_DIR / "data/test_cifar/test1.png"
# 与旧训练脚本默认保存位置一致；如使用其他权重，请修改此变量。
model_path = Path(__file__).resolve().parent / "models_cifar10/cifar7.pth"
if not model_path.exists():
    model_path = PROJECT_DIR / "models_cifar10/cifar7.pth"
with Image.open(img_path) as img:
    img = img.convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
model = cifar10()
model.load_state_dict(load_checkpoint_state(model_path))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
