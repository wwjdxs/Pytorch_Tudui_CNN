# 运行与复现指南

以下命令在仓库根目录运行。也可以在 IDE 选择对应 Python 环境后直接运行脚本。仓库主页只放项目说明，具体命令集中在本文。

## 1. 安装环境

```bash
python -m pip install -r requirements.txt
```

GPU 训练需要与本机驱动匹配的 CUDA 版 PyTorch，安装方式以 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)为准。历史实验环境是 PyTorch 2.15.0.dev20260913+cu130、RTX 5060 Laptop；本仓库不要求安装该开发版本，其他版本的结果可能存在数值差异。

只有旧教程的 TensorBoard 示例才需要额外安装：

```bash
python -m pip install -r requirements-tutorials.txt
```

## 2. 准备数据

当前入口默认读取仓库中的 Pytorch/data。目录下应有 cifar-10-batches-py；如果本地没有 CIFAR10，可运行一次下面的下载命令：

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('Pytorch/data', train=True, download=True)"
```

如果使用其他数据目录，在 PowerShell 设置环境变量即可，不需要同时修改两处代码：

```powershell
$env:CNN_DATA_DIR = "E:\你的数据目录"
```

也可直接修改脚本顶部的数据路径。TensorBoard 教程示例使用 Pytorch/data 中的对应图片，模型保存/加载示例按脚本所在目录定位权重。

## 3. 直接评估已发布的最终权重

```bash
python Pytorch/code/CIFAR_train_CNN/CNNtest.py
```

默认读取历史 09_loss_mean/best.pth，评估官方测试集。新版模型使用 ReLU、BatchNorm 和 Normalize((0.5,)*3, (0.5,)*3)，测试必须保持一致。CPU 可以运行评估，CUDA 可用时自动使用 GPU。

单图预测可在 CNNtest.py 顶部填写 IMAGE_PATH，或设置 CNN_IMAGE 环境变量；CNN_CHECKPOINT 环境变量可以指定自己的最终配置权重。此测试入口不自动猜测旧阶段的结构或标准化，旧权重应使用对应阶段的模型和预处理。

## 4. 训练最终配置

```bash
# GPU可用时自动使用GPU，否则回退CPU
python Pytorch/code/CIFAR_train_CNN/CNNtrain_gpu1.py

# 显式使用CPU；CPU训练较慢
python Pytorch/code/CIFAR_train_CNN/CNNtrain.py
```

两个入口共用 CNNmodel.py，使用相同划分和训练设置。新权重与逐轮指标写入当前CNN目录下带时间戳的 runs/final_* 目录，避免覆盖已发布实验。训练期间打印的是验证准确率；训练过程不会用官方测试集选模型。若测试新权重，请将 CNN_CHECKPOINT 指向该目录的 best.pth。

## 5. 完整复现逐步实验

```bash
python Pytorch/code/CIFAR_train_CNN/CNN_experiment_runner.py
```

完整实验使用 CUDA，需可用的 CUDA 环境。各阶段源码只生成在本轮 experiments/时间戳/阶段 目录内，不再覆盖主训练脚本。所有阶段结束后才统一测试。CPU 张量缓存减少重复 PIL 转换，随机增强仍每轮重新采样；数据目录统一由 CNN_DATA_DIR 控制。

运行结束后生成图表和分析，将下例时间戳替换为本轮的目录：

```bash
python Pytorch/code/CIFAR_train_CNN/CNN_experiment_analysis.py Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749
```

历史快照、指标和检查点记录当时的运行情况。当前入口的模型复用、路径兼容等修复不会回写历史文件，也没有因此重新训练历史的260轮实验。

## 6. 仓库自检

自检包含旧教程的TensorBoard示例，请先安装教程依赖：

```bash
python -m pip install -r requirements-tutorials.txt
python -m unittest discover -s tests -v
```

测试检查代码语法、Notebook代码、文档链接、不同阶段的权重兼容性，以及历史结果表与混淆矩阵的一致性，并用合成小数据运行教学示例。其他运行验证及未执行项目详见[仓库检查报告](REPOSITORY_AUDIT.md)。

## 旧教程说明

旧教程的 CIFAR_train 目录仍是无 ReLU/BatchNorm 的基线。先运行 train.py 或 train_gpu1.py 保存 state_dict，再运行 test.py；test.py 也可回退到仓库已有的基线权重。TensorBoard 的卷积、Linear 示例已修正图片维度与最后一个不足整批的问题。VGG 示例会下载较大权重，model_save.py 必须先于 model_load.py 运行；默认权重文件位置现在与脚本位置一致。
