# PyTorch 小土堆教程学习与 CIFAR10 CNN 实验

这个仓库记录了 B 站小土堆 PyTorch 教程的学习代码，以及在此基础上进行的 CNN 逐步改进实验。

> 如果你看到带有加号、减号和行号的页面，那是 GitHub 的提交差异页，显示的是 Markdown 源码。要阅读排版后的说明，请打开[仓库主页](https://github.com/wwjdxs/Pytorch_Tudui_CNN)或 README 的 Preview，而不是提交链接中的 diff。

## 当前结果

CNN 最终配置在 CIFAR10 官方测试集上的准确率为 **80.63%**。本轮固定种子 42，使用 45,000 张训练图片、5,000 张验证图片和 10,000 张官方测试图片；检查点只按验证准确率选择。10 个累计配置共训练 260 轮，测试准确率如下。

|累计配置|训练轮数|测试准确率|
|---|---:|---:|
|原始网络基线|10|44.89%|
|加 ReLU|10|35.92%|
|延长训练|30|64.72%|
|SGD 动量|30|65.71%|
|学习率衰减|30|70.10%|
|打乱训练数据|30|71.35%|
|标准化|30|74.97%|
|随机裁剪与水平翻转|30|79.57%|
|BatchNorm，并移除卷积 bias|30|80.63%|
|只修正 loss 显示口径|30|80.63%|

这是**单种子的探索性累加实验**，不能宣称统计显著，也不能断言每个改动对其他网络都有效。ReLU 在短训练中出现下降，最后两个配置的模型权重完全一致。早期使用全部 50,000 张训练图片得到的 56.25% 属于另一轮实验，不是本表的严格配对基线。

## 从哪里开始

- [运行与复现指南](docs/QUICKSTART.md)：环境准备、训练、测试和生成图表的具体步骤。
- [完整实验日志](Pytorch/code/CIFAR_train_CNN/CNN实验日志_20261016_150749.md)：逐步结果、代码改动、现象与局限。
- [仓库检查与修复报告](docs/REPOSITORY_AUDIT.md)：本次发现的问题、修复内容及验证范围。
- [结果总表](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/summary.csv)、[实验设置](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/protocol.json)、[最终最佳权重](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/09_loss_mean/best.pth)。

## 目录说明

|目录|用途|
|---|---|
|Pytorch/code/CIFAR_train_CNN|当前 CNN 模型、CPU/GPU 训练、测试和实验工具；推荐使用此目录|
|Pytorch/code/CIFAR_train_CNN/experiments|历史实验的源码快照、指标、图表与权重，不随当前代码修复而改写|
|Pytorch/code/CIFAR_train|早期教程基线，与最终 BatchNorm 模型不同|
|Pytorch/code/basic_tools|Dataset、DataLoader 和图像变换示例|
|Pytorch/code/torch.nn|卷积、池化、激活、Linear、损失与优化示例|
|Pytorch/code/model|模型保存加载、VGG 等示例|
|Pytorch/code/CNN_baseline_results|早期完整训练集基线记录|
|Pytorch/notebook|早期 Notebook 学习记录|

当前训练与测试入口共用 CNNmodel.py。新版 CNNtest.py 默认读取已发布的最终最佳权重，并采用对应的标准化。历史阶段快照仍保留当时的本机路径，应通过实验工具重新生成跨机器可运行的版本；不要直接将旧教程的模型类用于加载最终权重。

## 实验图表

![CNN逐步改进结果](Pytorch/code/CIFAR_train_CNN/experiments/20261016_150749/experiment_comparison.png)

## 数据与依赖

早期提交已经包含 CIFAR10 数据文件和 ants/bees 示例图片；.gitignore 不会自动取消对已提交文件的跟踪。当前版本保留这些教学数据；新产生的Python缓存和默认runs训练输出由.gitignore忽略，新的正式实验记录可按需要提交。历史中的数据仍会影响仓库下载大小，本次没有重写 Git 历史。

当前 CNN 入口不需要 TensorBoard。旧教程中的可视化示例需要额外安装 TensorBoard；VGG 预训练示例首次运行还需要下载官方权重，模型保存示例会生成较大的文件。具体环境与运行顺序见运行指南。

教程学习时间：2026 年 9 月至 10 月。原视频：[小土堆 PyTorch 教程](https://www.bilibili.com/video/BV1hE411t7RN/)。本仓库本轮不包含 ViT 实验。
