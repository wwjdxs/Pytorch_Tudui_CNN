"""逐步累加实验：生成当前训练脚本、运行、保存版本与结果。直接运行本文件即可复现。
研究辅助脚本集中使用CSV/JSON等记录工具，CNNtrain_gpu1.py仍保持顺序式教学结构。
"""
import os
import csv
import json
import runpy
import time
from pathlib import Path
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "CNNtrain_gpu1.py"
DATA = r"E:\DeepLearning\deep_learning_pytorch2.14\Pytorch\data"
SEED = 42
# 按预定顺序累加配置；增强为裁剪与翻转组合，BN步骤同时移除卷积bias。
STAGES = [
    ("00_baseline", 10, False, 0.0, False, False, False, False, False, False),
    ("01_relu", 10, True, 0.0, False, False, False, False, False, False),
    ("02_epochs30", 30, True, 0.0, False, False, False, False, False, False),
    ("03_momentum", 30, True, 0.9, False, False, False, False, False, False),
    ("04_scheduler", 30, True, 0.9, True, False, False, False, False, False),
    ("05_shuffle", 30, True, 0.9, True, True, False, False, False, False),
    ("06_normalize", 30, True, 0.9, True, True, True, False, False, False),
    ("07_augment", 30, True, 0.9, True, True, True, True, False, False),
    ("08_batchnorm", 30, True, 0.9, True, True, True, True, True, False),
    ("09_loss_mean", 30, True, 0.9, True, True, True, True, True, True),
]


def make_source(stage):
    name, epochs, relu, momentum, schedule, shuffle, normalize, augment, bn, mean_loss = stage
    layers = []
    for in_c, out_c in [(3, 32), (32, 32), (32, 64)]:
        layers.append(f"            nn.Conv2d({in_c}, {out_c}, kernel_size=5, padding=2, bias={not bn}),")
        if bn: layers.append(f"            nn.BatchNorm2d({out_c}),")
        if relu: layers.append("            nn.ReLU(),")
        layers.append("            nn.MaxPool2d(2),")
    layers += ["            nn.Flatten(),", "            nn.Linear(1024, 64),"]
    if relu: layers.append("            nn.ReLU(),")
    layers.append("            nn.Linear(64, 10)")
    train_transforms = []
    if augment:
        train_transforms += ["    torchvision.transforms.RandomCrop(32, padding=4),", "    torchvision.transforms.RandomHorizontalFlip(),"]
    train_transforms += ["    torchvision.transforms.ToTensor(),"]
    if normalize: train_transforms += ["    torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),"]
    eval_transforms = ["    torchvision.transforms.ToTensor(),"]
    if normalize: eval_transforms += ["    torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),"]
    return TEMPLATE.replace("__STAGE__",name).replace("__LAYERS__","\n".join(layers)).replace("__TRAIN_TRANSFORMS__","\n".join(train_transforms)).replace("__EVAL_TRANSFORMS__","\n".join(eval_transforms)).replace("__EPOCHS__",str(epochs)).replace("__MOMENTUM__",str(momentum)).replace("__SHUFFLE__",str(shuffle)).replace("__SCHEDULER_INIT__",'scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)' if schedule else '# 本阶段使用固定学习率。').replace("__SCHEDULER_STEP__",'    scheduler.step()' if schedule else '').replace("__DISPLAY_LOSS__",'validation_loss' if mean_loss else 'validation_batch_loss_sum')


# 模板保留与原CNN脚本相同的模型、数据、优化器、训练、评估、保存的阅读顺序。
TEMPLATE = r'''import os
import csv
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 实验阶段：__STAGE__。seed固定，使用同一训练/验证划分。
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
__LAYERS__
        )

    def forward(self, x):
        return self.model(x)

# 仅训练集应用随机增强；验证集使用确定性预处理。
data_path = r"E:\DeepLearning\deep_learning_pytorch2.14\Pytorch\data"
train_transform = torchvision.transforms.Compose([
__TRAIN_TRANSFORMS__
])
eval_transform = torchvision.transforms.Compose([
__EVAL_TRANSFORMS__
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
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=__SHUFFLE__,
                              generator=torch.Generator().manual_seed(42), pin_memory=device.type == "cuda")
validation_dataloader = DataLoader(validation_data, batch_size=256, shuffle=False,
                                   pin_memory=device.type == "cuda")

cifar = cifar10().to(device)
loss_fun = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(cifar.parameters(), lr=learning_rate, momentum=__MOMENTUM__)
__SCHEDULER_INIT__
epoch = __EPOCHS__
current_path = os.path.dirname(os.path.abspath(__file__))
# 批量实验器可指定隔离的输出目录，直接运行时使用本阶段默认目录。
save_path = os.environ.get("CNN_EXPERIMENT_OUTPUT", os.path.join(current_path, "experiments", "__STAGE__"))
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
    print("__STAGE__ {}/{} train={:.2%} val={:.2%} loss={:.4f}".format(
        i+1, epoch, row["train_accuracy"], validation_accuracy, __DISPLAY_LOSS__), flush=True)
__SCHEDULER_STEP__
torch.save(cifar.state_dict(), os.path.join(save_path, "last.pth"))
training_seconds = time.perf_counter()-start
print("__STAGE__ 完成，最佳验证轮次：{}，耗时：{:.1f}s".format(best_epoch, training_seconds), flush=True)
'''


class CachedCIFAR(Dataset):
    """缓存等价的ToTensor结果，避免每轮重复PIL转换；不缓存随机增强。"""
    cache = {}
    original = torchvision.datasets.CIFAR10

    def __init__(self, root, train=True, download=False, transform=None, **kwargs):
        if train not in self.cache:
            data = self.original(root, train=train, download=download)
            tensor = torch.from_numpy(data.data).permute(0,3,1,2).contiguous().float().div_(255)
            self.cache[train] = (tensor, data.targets, data.classes)
        self.images, self.targets, self.classes = self.cache[train]
        # 裁剪、翻转、标准化对Tensor与PIL转换后的像素运算等价。
        self.transforms = [] if transform is None else [t for t in transform.transforms if not isinstance(t, torchvision.transforms.ToTensor)]

    def __len__(self): return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        for transform in self.transforms: image = transform(image)
        return image, self.targets[index]


def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(10,10,dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            confusion += torch.bincount((labels*10+predictions).cpu(), minlength=100).reshape(10,10)
    return dict(accuracy=correct/total, mean_loss=loss_sum/total, confusion=confusion.tolist())


def main():
    if not torch.cuda.is_available(): raise RuntimeError("本实验预设使用CUDA")
    torch.set_num_threads(4)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = BASE / "experiments" / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output/"original_CNNtrain_gpu1.py").write_bytes(SOURCE.read_bytes())
    metadata = dict(seed=SEED, train_samples=45000, validation_samples=5000, test_samples=10000,
                    device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
                    stages=STAGES, batch_size=64, evaluation_batch_size=256,
                    deterministic=True, cache="CPU ToTensor cache, no random augmentation cached")
    (output/"protocol.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    log = BASE / ("CNN实验日志_"+stamp+".md")
    log.write_text("# CNN逐步改进实验日志\n\n固定种子42，分层训练/验证=45000/5000；测试集10000张在全部方案完成后统一评估。每步从头训练，采用验证准确率最高的轮次。单种子探索性实验。\n\n输出目录：`"+str(output)+"`\n\n",encoding="utf-8")
    print("RUN_DIRECTORY:",output,flush=True)
    print("LOG_FILE:",log,flush=True)
    torchvision.datasets.CIFAR10 = CachedCIFAR
    summaries = []
    # 依次生成、修改、运行、记录，禁止按测试集结果改变后续预先指定的方案。
    for stage in STAGES:
        name = stage[0]
        stage_dir = output/name
        stage_dir.mkdir()
        source = make_source(stage)
        SOURCE.write_text(source,encoding="utf-8")
        (stage_dir/"CNNtrain_gpu1.py").write_text(source,encoding="utf-8")
        os.environ["CNN_EXPERIMENT_OUTPUT"] = str(stage_dir)
        namespace = runpy.run_path(str(SOURCE),run_name="__main__")
        best = max(namespace["history"],key=lambda row: row["validation_accuracy"])
        summary = dict(stage=name, epochs=stage[1], best_epoch=best["epoch"],
                       validation_accuracy=best["validation_accuracy"], validation_loss=best["validation_loss"],
                       last_validation_accuracy=namespace["history"][-1]["validation_accuracy"],
                       training_seconds=namespace["training_seconds"],
                       parameters=sum(p.numel() for p in namespace["cifar"].parameters()))
        summaries.append(summary)
        (output/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
        with log.open("a",encoding="utf-8") as file:
            file.write("## "+name+"\n\n训练轮数：{}；最佳验证轮次：{}；验证准确率：{:.2%}；验证平均loss：{:.4f}；训练耗时：{:.1f}s；参数量：{}。\n\n".format(stage[1],best["epoch"],best["validation_accuracy"],best["validation_loss"],summary["training_seconds"],summary["parameters"]))
        # 原始模型和ReLU均有固定初始化；延长轮数前10轮应完全复现短训练。
        if name == "02_epochs30":
            prior = list(csv.DictReader((output/"01_relu"/"metrics.csv").open()))
            assert all(float(prior[i]["validation_accuracy"]) == namespace["history"][i]["validation_accuracy"] for i in range(10))
        del namespace
        torch.cuda.empty_cache()
    # 最后两阶段只改变loss的展示方式，所有参数应逐元素完全一致。
    first = torch.load(output/"08_batchnorm"/"last.pth",map_location="cpu",weights_only=True)
    second = torch.load(output/"09_loss_mean"/"last.pth",map_location="cpu",weights_only=True)
    same_weights = all(torch.equal(first[key],second[key]) for key in first)
    print("LOSS_REPORTING_WEIGHTS_IDENTICAL:",same_weights,flush=True)
    with log.open("a",encoding="utf-8") as file:
        file.write("loss展示修正前后最终权重完全一致：`"+str(same_weights)+"`。\n\n")
    # 所有配置已执行完毕，现在才评估各版本最佳验证权重的官方测试集表现。
    for stage, summary in zip(STAGES,summaries):
        name = stage[0]
        # 只执行模型与数据定义，避免再次启动训练。
        source = make_source(stage).split("cifar = cifar10().to(device)")[0]
        namespace = {"__file__":str(SOURCE)}
        exec(compile(source,str(SOURCE),"exec"),namespace)
        model = namespace["cifar10"]().cuda()
        model.load_state_dict(torch.load(output/name/"best.pth",weights_only=True))
        test_data = CachedCIFAR(DATA,train=False,transform=namespace["eval_transform"])
        clean_train = torch.utils.data.Subset(namespace["validation_all"],namespace["train_indices"])
        train_metrics = evaluate(model,DataLoader(clean_train,batch_size=256))
        test_metrics = evaluate(model,DataLoader(test_data,batch_size=256))
        summary.update(clean_train_accuracy=train_metrics["accuracy"],clean_train_loss=train_metrics["mean_loss"],
                       test_accuracy=test_metrics["accuracy"],test_loss=test_metrics["mean_loss"])
        (output/name/"evaluation.json").write_text(json.dumps(dict(train=train_metrics,test=test_metrics),indent=2),encoding="utf-8")
        (output/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
        print("FINAL_EVAL",name,summary["clean_train_accuracy"],summary["test_accuracy"],flush=True)
        with log.open("a",encoding="utf-8") as file:
            file.write("- {}：无增强训练集准确率 {:.2%}，官方测试准确率 {:.2%}，测试平均loss {:.4f}。\n".format(name,train_metrics["accuracy"],test_metrics["accuracy"],test_metrics["mean_loss"]))
        del model,namespace
    with (output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as file:
        recorder=csv.DictWriter(file,fieldnames=list(summaries[0]))
        recorder.writeheader(); recorder.writerows(summaries)
    os.environ.pop("CNN_EXPERIMENT_OUTPUT",None)
    print("ALL_EXPERIMENTS_COMPLETE",output,flush=True)


if __name__ == "__main__":
    main()
