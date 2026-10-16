"""当前CNN模型。默认结构与历史08/09阶段权重完全一致。"""
import torch
import torch.nn as nn


class cifar10(nn.Module):
    def __init__(self, relu=True, batchnorm=True):
        super().__init__()
        # 历史基线可用cifar10(relu=False, batchnorm=False)重建。
        layers = []
        for in_channels, out_channels in ((3, 32), (32, 32), (32, 64)):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5,
                                    padding=2, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            if relu:
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
        layers.extend([nn.Flatten(), nn.Linear(1024, 64)])
        if relu:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 输出[B,10]原始logits，交叉熵直接接收，不在模型中做softmax。
        return self.model(x)


if __name__ == "__main__":
    model = cifar10()
    print(model(torch.ones(2, 3, 32, 32)).shape)
