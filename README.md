2026/9-2026/10/11

============完成b站小土堆Pytorch视频跟练=============

视频链接：https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=652256353e7af95b93d5bf06f3150d94

全程手打代码+根据版本更新修改了部分代码

最终产出：CIFAR10分类模型，但是正确率较低，没有训练完成


## 2026-10-16：CNN逐步改进实验

在原教程代码之外新增 [CNN实验目录](Pytorch/code/CIFAR_train_CNN)，保留早期 `Pytorch/code/CIFAR_train` 作为学习记录。本次未包含ViT代码。

固定随机种子42，将CIFAR10官方训练集分层划分为45,000张训练、5,000张验证。各阶段从头训练，按验证准确率选择检查点；全部方案完成后统一评估官方10,000张测试图片，共10组、260轮训练。

|累计改动|训练轮数|测试准确率|
|---|---:|---:|
|原始网络基线|10|44.89%|
|加ReLU|10|35.92%|
|延长训练|30|64.72%|
|SGD动量|30|65.71%|
|学习率衰减|30|70.10%|
|打乱训练数据|30|71.35%|
|标准化|30|74.97%|
|随机裁剪和水平翻转|30|79.57%|
|BatchNorm，并移除卷积bias|30|80.63%|
|仅修正loss显示口径|30|80.63%|

最终比本轮基线提高35.74个百分点。参数量从145,578增至145,706。数据增强前后，无增强训练集与测试集的准确率差距从24.05降至3.58个百分点。loss显示修正前后的模型权重完全一致。

这是单种子的探索性累加实验，不能宣称统计显著；ReLU短训练出现下降，不能声称每一步都能提升准确率。BN步骤同时移除了卷积bias。各阶段效应依赖此前配置。此前使用全部50,000张训练图片得到的56.25%不是本轮严格配对基线。

### 日志与结果

- [完整实验日志与分析](Pytorch/code/CIFAR_train_CNN/CNN实验日志_20261016_150749.md)
- [实验结果CSV](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/summary.csv)
- [实验设置](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/protocol.json)
- [逐步代码差异](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/changes.patch)
- [最终最佳权重](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/09_loss_mean/best.pth)
- [原始脚本备份](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/original_CNNtrain_gpu1.py)
- [早期完整训练集基线记录](Pytorch/code/CNN_baseline_results/20261016_145714)

![逐步实验图表](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/experiment_comparison.png)

### 复现

训练环境需要PyTorch、torchvision和Pillow；图表分析另外需要matplotlib。本次运行PyTorch 2.15.0.dev20260913+cu130，GPU为RTX 5060 Laptop，版本信息保存在protocol.json。无需TensorBoard。

先下载CIFAR10，并修改 `CNN_experiment_runner.py` 中的 `DATA` 和模板中的 `data_path` 为本机数据目录；直接运行最终训练脚本时，仅需修改该脚本的 `data_path`。实验器包含CUDA专用评估，本轮完整复现需要可用的CUDA环境。

```bash
# 逐步生成各版本、训练并记录，结果进入带时间戳的experiments子目录
python Pytorch/code/CIFAR_train_CNN/CNN_experiment_runner.py

# 读取已完成的结果，生成分析日志与图表（将时间戳替换为自己的运行目录）
python Pytorch/code/CIFAR_train_CNN/CNN_experiment_analysis.py Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749

# 只训练最终配置
python Pytorch/code/CIFAR_train_CNN/CNNtrain_gpu1.py
```

每阶段包含当时的源码、逐轮指标、最佳/最终权重和混淆矩阵。当前 `CNNtrain_gpu1.py` 是最终阶段的版本。实验器会依次重写该脚本，请先保存个人修改。完整实验器用等价的CPU张量缓存减少重复PIL转换；每轮随机增强不会被缓存。仓库不包含数据集、Python缓存或本次TensorBoard临时文件。日志中的绝对路径是原实验机器的历史记录，下载后以仓库相对路径为准。
