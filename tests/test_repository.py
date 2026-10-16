"""仓库回归检查：不下载权重、不启动完整训练、不重写历史实验。"""
import ast
import csv
import importlib.util
import json
import os
import re
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CNN = ROOT / "Pytorch/code/CIFAR_train_CNN"
RUN = CNN / "experiments/20261016_150749"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.paths = [ROOT / p for p in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if p]
        cls.model_module = load_module("cnn_model_for_audit", CNN / "CNNmodel.py")
        cls.legacy_module = load_module("legacy_model_for_audit", ROOT / "Pytorch/code/CIFAR_train/CNNmodel.py")
        cls.runner = load_module("cnn_runner_for_audit", CNN / "CNN_experiment_runner.py")

    def test_all_python_syntax(self):
        for p in self.paths + list((ROOT/"tests").glob("*.py")):
            if p.suffix == ".py":
                with self.subTest(file=str(p.relative_to(ROOT))):
                    ast.parse(p.read_text(encoding="utf-8-sig"))

    def test_notebooks_are_valid_json(self):
        for p in self.paths:
            if p.suffix == ".ipynb":
                notebook = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                self.assertIsInstance(notebook["cells"], list)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell.get("source", [])))

    def test_local_markdown_links_and_fences(self):
        for p in list(ROOT.rglob("*.md")):
            if ".git" in p.parts:
                continue
            text = p.read_text(encoding="utf-8-sig")
            self.assertEqual(sum(line.startswith("```") for line in text.splitlines()) % 2, 0, p)
            for link in re.findall(r"\]\(([^)]+)\)", text):
                if link.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = p.parent / link.split("#")[0]
                self.assertTrue(target.exists(), "{} -> {}".format(p, link))

    def test_current_model_loads_every_stage(self):
        for stage in self.runner.STAGES:
            name, _, relu, _, _, _, _, _, bn, _ = stage
            model = self.model_module.cifar10(relu=relu, batchnorm=bn).eval()
            for checkpoint in ("best.pth", "last.pth"):
                with self.subTest(stage=name, checkpoint=checkpoint):
                    state = torch.load(RUN/name/checkpoint, map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                    with torch.no_grad():
                        output = model(torch.zeros(2, 3, 32, 32))
                    self.assertEqual(tuple(output.shape), (2, 10))
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_all_published_weights_are_finite(self):
        for p in self.paths:
            if p.suffix != ".pth":
                continue
            with self.subTest(checkpoint=str(p.relative_to(ROOT))):
                state = self.legacy_module.load_checkpoint_state(p) if p.parent == ROOT/"Pytorch/models_cifar10" else torch.load(p, map_location="cpu", weights_only=True)
                self.assertIsInstance(state, dict)
                self.assertTrue(state)
                self.assertTrue(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
                                    for v in state.values()))

    def test_recorded_results_are_consistent(self):
        rows = json.loads((RUN/"summary.json").read_text())
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(row["epochs"] for row in rows), 260)
        for row in rows:
            directory = RUN/row["stage"]
            with (directory/"metrics.csv").open() as file:
                history = list(csv.DictReader(file))
            self.assertEqual(len(history), row["epochs"])
            best = max(history, key=lambda r: float(r["validation_accuracy"]))
            self.assertEqual(int(best["epoch"]), row["best_epoch"])
            self.assertEqual(float(best["validation_accuracy"]), row["validation_accuracy"])
            metrics = json.loads((directory/"evaluation.json").read_text())
            confusion = metrics["test"]["confusion"]
            self.assertTrue(all(sum(r) == 1000 for r in confusion))
            accuracy = sum(confusion[i][i] for i in range(10))/10000
            self.assertEqual(accuracy, row["test_accuracy"])

    def test_loss_reporting_does_not_change_weights(self):
        for checkpoint in ("best.pth", "last.pth"):
            a = torch.load(RUN/"08_batchnorm"/checkpoint, map_location="cpu", weights_only=True)
            b = torch.load(RUN/"09_loss_mean"/checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(set(a), set(b))
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_generated_stages_use_one_data_directory(self):
        for stage in self.runner.STAGES:
            source = self.runner.make_source(stage)
            ast.parse(source)
            self.assertNotIn("__DATA_PATH__", source)
            self.assertIn(repr(str(self.runner.DATA)), source)
        with patch.object(self.runner, "SEED", 7):
            altered = self.runner.make_source(self.runner.STAGES[0])
            self.assertIn("manual_seed(7)", altered)
            self.assertNotIn("manual_seed(42)", altered)
        self.assertNotIn("SOURCE.write_text", (CNN/"CNN_experiment_runner.py").read_text(encoding="utf-8"))

    def test_runner_restores_state_after_failure(self):
        original_dataset = self.runner.torchvision.datasets.CIFAR10
        original_output = os.environ.get("CNN_EXPERIMENT_OUTPUT")
        def fail():
            self.runner.torchvision.datasets.CIFAR10 = object
            os.environ["CNN_EXPERIMENT_OUTPUT"] = "temporary-output"
            raise RuntimeError("intentional failure for cleanup test")
        with patch.object(self.runner, "_run_experiments", fail):
            with self.assertRaises(RuntimeError):
                self.runner.main()
        self.assertIs(self.runner.torchvision.datasets.CIFAR10, original_dataset)
        self.assertEqual(os.environ.get("CNN_EXPERIMENT_OUTPUT"), original_output)

    def test_cached_datasets_do_not_mix_roots(self):
        class TinyRaw:
            def __init__(self, root, **kwargs):
                self.data = np.full((2,32,32,3), 10 if str(root).endswith("first") else 20, dtype=np.uint8)
                self.targets, self.classes = [0,0], ["airplane"]
        cache = self.runner.CachedCIFAR
        cache.cache.clear()
        try:
            with patch.object(cache, "original", TinyRaw):
                first = cache(ROOT/"first")
                second = cache(ROOT/"second")
            self.assertFalse(torch.equal(first[0][0], second[0][0]))
            self.assertEqual(len(cache.cache), 2)
        finally:
            cache.cache.clear()

    def test_forward_backward_and_short_batch(self):
        model = self.model_module.cifar10()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.model[-1].weight.detach().clone()
        # 模拟不足一个batch的3张图，验证分类头不依赖固定batch大小。
        output = model(torch.rand(3, 3, 32, 32))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.model[-1].weight))


if __name__ == "__main__":
    unittest.main()
