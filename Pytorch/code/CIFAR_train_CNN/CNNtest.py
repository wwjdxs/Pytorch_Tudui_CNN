"""评估最终CNN权重，或预测一张图片。导入本文件不会触发训练或评估。"""
import os
from pathlib import Path
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from CNNmodel import cifar10

BASE_DIR = Path(__file__).resolve().parent
# 可直接修改这三个变量，也可用同名环境变量指定路径。
DATA_DIR = Path(os.environ.get("CNN_DATA_DIR", BASE_DIR.parents[1] / "data"))
CHECKPOINT = Path(os.environ.get("CNN_CHECKPOINT", BASE_DIR / "experiments" /
    "20261016_150749" / "09_loss_mean" / "best.pth"))
IMAGE_PATH = os.environ.get("CNN_IMAGE", "")  # 留空评估测试集；填入路径则做单图预测。


def get_transform():
    # 必须与最终训练设置一致。评估不使用随机裁剪和翻转。
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])


def load_model(checkpoint=CHECKPOINT, device="cpu"):
    model = cifar10()
    # 保存的是state_dict，先创建模型，再加载参数；映射到CPU兼容GPU保存的权重。
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    if IMAGE_PATH:
        with Image.open(IMAGE_PATH) as image:
            image = image.convert("RGB").resize((32, 32))
            image = get_transform()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(image).argmax(1).item()
        classes = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
        print("预测类别：{} ({})".format(classes[prediction], prediction))
        return

    test_data = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False,
                                            transform=get_transform())
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, targets in DataLoader(test_data, batch_size=256):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(1) == targets).sum().item()
    print("测试样本：{}，准确率：{:.2%}，平均loss：{:.4f}".format(
        len(test_data), correct / len(test_data), total_loss / len(test_data)))


if __name__ == "__main__":
    main()
