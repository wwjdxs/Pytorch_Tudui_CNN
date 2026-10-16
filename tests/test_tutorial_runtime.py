"""用65张合成图片检查教学脚本的维度，避免下载和长训练。"""
import ast
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class TinyCIFAR:
    def __init__(self, root, train=False, transform=None, **kwargs):
        self.transform = transform
        self.data = np.zeros((65, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(65)]
        self.classes = [str(i) for i in range(10)]

    def __len__(self):
        return 65

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform:
            image = self.transform(image)
        return image, self.targets[index]


class WriterSpy:
    opened = []

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.opened.append(self)

    def add_image(self, tag, tensor, *args, **kwargs):
        if tensor.ndim != 3:
            raise AssertionError("add_image需要3维CHW")

    def add_images(self, tag, tensor, *args, **kwargs):
        if tensor.ndim != 4:
            raise AssertionError("add_images需要4维NCHW")

    def add_histogram(self, tag, tensor, *args, **kwargs):
        if not torch.isfinite(tensor).all():
            raise AssertionError("直方图包含非有限数值")

    def add_text(self, *args, **kwargs):
        pass

    def add_graph(self, model, inputs, *args, **kwargs):
        model(inputs)

    def close(self):
        self.closed = True


class TutorialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_example(self, relative_path):
        p = ROOT / relative_path
        # 仅缩短示例循环；文件本身和教学参数不改写。
        source = p.read_text(encoding="utf-8").replace("range(10)", "range(1)")
        namespace = {"__file__":str(p), "__name__":"__main__"}
        WriterSpy.opened = []
        with patch("torchvision.datasets.CIFAR10", TinyCIFAR), patch(
                "torch.utils.tensorboard.SummaryWriter", WriterSpy), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, str(p), "exec"), namespace)
        self.assertTrue(all(writer.closed for writer in WriterSpy.opened), p)
        return namespace

    def test_linear_last_batch_is_one_image(self):
        namespace = self.run_example("Pytorch/code/torch.nn/nn_linear.py")
        self.assertEqual(tuple(namespace["output"].shape), (1, 10))

    def test_conv_batch_visualization(self):
        namespace = self.run_example("Pytorch/code/torch.nn/nn_conv2d.py")
        self.assertEqual(namespace["output"].ndim, 4)

    def test_tensor_and_network_examples(self):
        for name in ("convolution.py", "lossfunction.py", "maxpool2d.py", "nn_RELU.py",
                     "nn_Sigmoid.py", "nn_Module.py", "nn_seq.py", "nn_loss_network.py", "nn_optim.py"):
            with self.subTest(example=name):
                self.run_example("Pytorch/code/torch.nn/"+name)

    def test_dataset_and_transform_examples(self):
        for name in ("dataloader.py", "dataset&dataloader.py", "Transforms.py",
                     "commonly_used_Transforms.py", "dataset_transforms.py"):
            with self.subTest(example=name):
                self.run_example("Pytorch/code/basic_tools/"+name)

    def test_vgg_append_layer_accepts_1000_logits(self):
        p = ROOT/"Pytorch/code/model/vgg16.py"
        tree = ast.parse(p.read_text(encoding="utf-8"))
        append = next(node for node in tree.body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == "add_module")
        # 只验证分类头维度，不实例化大VGG、不下载预训练权重。
        class Head:
            classifier = torch.nn.Sequential(torch.nn.Linear(4096, 1000))
        namespace = {"vgg16_true":Head(), "nn":torch.nn}
        exec(compile(ast.Module(body=[append], type_ignores=[]), str(p), "exec"), namespace)
        self.assertEqual(tuple(namespace["vgg16_true"].classifier(torch.rand(2,4096)).shape), (2,10))


if __name__ == "__main__":
    unittest.main()
