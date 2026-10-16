import os
import csv
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 当前最终配置：seed固定，使用与历史实验相同的训练/验证划分。
# 历史09阶段源码保存在experiments中；本入口只修复路径与共享模型，不改变网络设置。
device = torch.device("cpu")  # CPU版与GPU版使用同一模型、数据处理和训练设置。
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# 对CUDA提供确定性矩阵乘法设置，比较两种loss统计时可检查权重完全一致。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# 与测试脚本共用模型定义，避免训练和加载结构不一致。
from CNNmodel import cifar10

# 仅训练集应用随机增强；验证集使用确定性预处理。
data_path = os.environ.get("CNN_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")))
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),
])
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),
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
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,
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
# 每次直接运行使用新的runs目录，避免覆盖已发布的历史实验。
save_path = os.environ.get("CNN_EXPERIMENT_OUTPUT", os.path.join(current_path, "runs", "final_" + time.strftime("%Y%m%d_%H%M%S")))
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
    print("09_loss_mean {}/{} train={:.2%} val={:.2%} loss={:.4f}".format(
        i+1, epoch, row["train_accuracy"], validation_accuracy, validation_loss), flush=True)
    scheduler.step()
torch.save(cifar.state_dict(), os.path.join(save_path, "last.pth"))
training_seconds = time.perf_counter()-start
print("09_loss_mean 完成，最佳验证轮次：{}，耗时：{:.1f}s".format(best_epoch, training_seconds), flush=True)
