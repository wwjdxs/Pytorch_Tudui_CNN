import os
import csv
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 实验阶段：04_scheduler。seed固定，使用同一训练/验证划分。
# 直接运行本文件会训练本阶段；批量逐步实验请运行CNN_experiment_runner.py。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# 对CUDA提供确定性矩阵乘法设置，比较两种loss统计时可检查权重完全一致。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

class cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# 仅训练集应用随机增强；验证集使用确定性预处理。
data_path = r"E:\DeepLearning\deep_learning_pytorch2.14\Pytorch\data"
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_all = torchvision.datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform)
validation_all = torchvision.datasets.CIFAR10(data_path, train=True, download=False, transform=eval_transform)
# 按类别分层抽取，每类4500张训练、500张验证，所有阶段使用完全相同的索引。
generator = torch.Generator().manual_seed(42)
train_indices, validation_indices = [], []
labels = torch.tensor(train_all.targets)
for label in range(10):
    indices = torch.where(labels == label)[0]
    indices = indices[torch.randperm(len(indices), generator=generator)]
    validation_indices.extend(indices[:500].tolist())
    train_indices.extend(indices[500:].tolist())
# 恢复原数据顺序；shuffle阶段才在每轮随机打乱。
train_indices.sort()
validation_indices.sort()
train_data = Subset(train_all, train_indices)
validation_data = Subset(validation_all, validation_indices)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False,
                              generator=torch.Generator().manual_seed(42), pin_memory=device.type == "cuda")
validation_dataloader = DataLoader(validation_data, batch_size=256, shuffle=False,
                                   pin_memory=device.type == "cuda")

cifar = cifar10().to(device)
loss_fun = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(cifar.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epoch = 30
current_path = os.path.dirname(os.path.abspath(__file__))
# 批量实验器可指定隔离的输出目录，直接运行时使用本阶段默认目录。
save_path = os.environ.get("CNN_EXPERIMENT_OUTPUT", os.path.join(current_path, "experiments", "04_scheduler"))
os.makedirs(save_path, exist_ok=True)
best_validation_accuracy = -1
history = []
if device.type == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()

for i in range(epoch):
    epoch_start = time.perf_counter()
    cifar.train()
    total_train_loss, total_train_correct = 0.0, 0
    for imgs, target in train_dataloader:
        imgs, target = imgs.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = cifar(imgs)
        loss = loss_fun(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * imgs.size(0)
        total_train_correct += (output.argmax(1) == target).sum().item()

    cifar.eval()
    total_validation_loss, total_correct_count = 0.0, 0
    validation_batch_loss_sum = 0.0
    with torch.no_grad():
        for imgs, target in validation_dataloader:
            imgs, target = imgs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = cifar(imgs)
            loss = loss_fun(output, target)
            total_validation_loss += loss.item() * imgs.size(0)
            validation_batch_loss_sum += loss.item()
            total_correct_count += (output.argmax(1) == target).sum().item()
    validation_loss = total_validation_loss / len(validation_data)
    validation_accuracy = total_correct_count / len(validation_data)
    row = dict(epoch=i+1, lr=optimizer.param_groups[0]["lr"],
               train_loss=total_train_loss/len(train_data), train_accuracy=total_train_correct/len(train_data),
               validation_loss=validation_loss, validation_accuracy=validation_accuracy,
               legacy_loss_sum=validation_batch_loss_sum, seconds=time.perf_counter()-epoch_start)
    history.append(row)
    with open(os.path.join(save_path, "metrics.csv"), "w", newline="", encoding="utf-8") as file:
        recorder = csv.DictWriter(file, fieldnames=list(row))
        recorder.writeheader()
        recorder.writerows(history)
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(cifar.state_dict(), os.path.join(save_path, "best.pth"))
        best_epoch = i+1
    # 前面阶段展示原始batch均值之和；最后阶段只改展示为样本平均loss。
    # CSV始终同时记录两者，便于比较；该改变不参与反向传播。
    print("04_scheduler {}/{} train={:.2%} val={:.2%} loss={:.4f}".format(
        i+1, epoch, row["train_accuracy"], validation_accuracy, validation_batch_loss_sum), flush=True)
    scheduler.step()
torch.save(cifar.state_dict(), os.path.join(save_path, "last.pth"))
training_seconds = time.perf_counter()-start
print("04_scheduler 完成，最佳验证轮次：{}，耗时：{:.1f}s".format(best_epoch, training_seconds), flush=True)
