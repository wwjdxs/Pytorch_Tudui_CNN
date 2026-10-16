"""读取已完成的逐步实验，生成图表并向同目录Markdown日志追加分析。
使用安装了matplotlib的Python执行：python CNN_experiment_analysis.py 实验目录。
"""
import csv
import difflib
import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Baseline 10ep", "+ ReLU 10ep", "+ 30 epochs", "+ Momentum",
          "+ LR decay", "+ Shuffle", "+ Normalize", "+ Augment", "+ BatchNorm", "+ Mean loss log"]
CHANGES = ["原始网络，10轮", "卷积和隐藏Linear后加ReLU", "训练10→30轮", "SGD动量0→0.9",
           "StepLR每10轮乘0.1", "训练shuffle=False→True", "输入标准化至约[-1,1]",
           "随机裁剪+水平翻转", "BatchNorm并移除卷积bias", "仅将显示loss改为样本平均"]
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def wilson(p, n=10000):
    z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius,center+radius


def main():
    output=Path(sys.argv[1]).resolve()
    rows=json.loads((output/"summary.json").read_text())
    assert len(rows)==10 and all("test_accuracy" in row for row in rows), "实验尚未全部完成"
    changes = []
    for previous, current in zip(rows, rows[1:]):
        old_source = (output/previous["stage"]/"CNNtrain_gpu1.py").read_text(encoding="utf-8").splitlines(keepends=True)
        new_source = (output/current["stage"]/"CNNtrain_gpu1.py").read_text(encoding="utf-8").splitlines(keepends=True)
        changes.extend(difflib.unified_diff(old_source, new_source, fromfile=previous["stage"], tofile=current["stage"]))
    (output/"changes.patch").write_text("".join(changes), encoding="utf-8")
    histories=[list(csv.DictReader((output/row["stage"]/"metrics.csv").open())) for row in rows]
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout="constrained")
    for index,(row,history) in enumerate(zip(rows,histories)):
        if index==9: continue
        axes[0,0].plot([int(h["epoch"]) for h in history],
                       [float(h["validation_accuracy"])*100 for h in history],label=LABELS[index])
    axes[0,0].set(xlabel="Epoch",ylabel="Validation accuracy (%)",title="Validation learning curves")
    axes[0,0].legend(fontsize=8,ncol=2)
    x=list(range(len(rows)))
    axes[0,1].bar([v-.18 for v in x],[r["validation_accuracy"]*100 for r in rows],.36,label="Best validation")
    axes[0,1].bar([v+.18 for v in x],[r["test_accuracy"]*100 for r in rows],.36,label="Held-out test")
    axes[0,1].set_xticks(x,LABELS,rotation=45,ha="right",fontsize=8)
    axes[0,1].set(ylabel="Accuracy (%)",title="Predefined cumulative interventions (one seed)")
    axes[0,1].legend()
    axes[1,0].bar(x,[(r["clean_train_accuracy"]-r["test_accuracy"])*100 for r in rows])
    axes[1,0].set_xticks(x,LABELS,rotation=45,ha="right",fontsize=8)
    axes[1,0].set(ylabel="Train minus test (percentage points)",title="Generalization gap at validation-selected epoch")
    axes[1,1].scatter([r["training_seconds"] for r in rows],[r["test_accuracy"]*100 for r in rows])
    for i,r in enumerate(rows): axes[1,1].annotate(str(i),(r["training_seconds"],r["test_accuracy"]*100),fontsize=9)
    axes[1,1].set(xlabel="Training + validation + checkpoint time (s)",ylabel="Test accuracy (%)",title="Observed accuracy / runtime trade-off")
    fig.savefig(output/"experiment_comparison.png",dpi=180)
    fig.savefig(output/"experiment_comparison.pdf")
    plt.close(fig)
    first=json.loads((output/rows[0]["stage"]/"evaluation.json").read_text())
    final=json.loads((output/rows[-1]["stage"]/"evaluation.json").read_text())
    recalls=[]
    for evaluation in (first,final):
        matrix=evaluation["test"]["confusion"]
        recalls.append([matrix[i][i]/sum(matrix[i]) for i in range(10)])
    fig,ax=plt.subplots(figsize=(11,4),layout="constrained")
    ax.bar([i-.2 for i in range(10)],[r*100 for r in recalls[0]],.4,label="Baseline")
    ax.bar([i+.2 for i in range(10)],[r*100 for r in recalls[1]],.4,label="Final")
    ax.set_xticks(range(10),CLASSES,rotation=25)
    ax.set(ylabel="Class recall (%)",title="CIFAR10 per-class recall: baseline vs final")
    ax.set_ylim(0, 100)
    ax.legend(ncol=2)
    fig.savefig(output/"class_recall.png",dpi=180)
    plt.close(fig)
    lines=["\n\n# 实测汇总与分析\n", "所有差值均为百分点，评估使用每个阶段验证准确率最高的检查点。\n",
           "|阶段|改动|轮数|最佳轮|验证准确率|测试准确率|无增强训练准确率|测试较上步变化|耗时(s)|参数量|",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i,r in enumerate(rows):
        delta="—" if i==0 else "{:+.2f}".format((r["test_accuracy"]-rows[i-1]["test_accuracy"])*100)
        lines.append("|{}|{}|{}|{}|{:.2%}|{:.2%}|{:.2%}|{}|{:.1f}|{}|".format(r["stage"],CHANGES[i],r["epochs"],r["best_epoch"],r["validation_accuracy"],r["test_accuracy"],r["clean_train_accuracy"],delta,r["training_seconds"],r["parameters"]))
    lines += ["\n## 收敛速度观察\n", "阈值仅用于描述已记录的曲线，不用于改变配置或选择检查点。", "|阶段|首次验证准确率达到70%的轮数|截至该轮的累计耗时(s)|", "|---|---:|---:|"]
    for row, history in zip(rows, histories):
        reached = next((i for i,h in enumerate(history) if float(h["validation_accuracy"]) >= 0.70), None)
        if reached is None:
            lines.append("|{}|本预算内未达到|—|".format(row["stage"]))
        else:
            lines.append("|{}|{}|{:.1f}|".format(row["stage"], reached+1, sum(float(h["seconds"]) for h in history[:reached+1])))
    lines += ["\n## 可以由本次实验支持的观察\n"]
    lines.append("1. **激活函数与训练预算存在条件依赖。** ReLU的10轮验证准确率由{:.2%}变为{:.2%}，其30轮达到{:.2%}。因此本次不能声称加ReLU就会立刻提高准确率；前10轮逐轮复现已核对。由于没有无ReLU的30轮对照，不能声称ReLU在30轮优于原网络。".format(rows[0]["validation_accuracy"],rows[1]["validation_accuracy"],rows[2]["validation_accuracy"]))
    for i,title in [(3,"动量"),(4,"学习率衰减"),(5,"打乱顺序"),(6,"标准化"),(7,"数据增强"),(8,"BatchNorm配置")]:
        r,previous=rows[i],rows[i-1]
        lines.append("{}. **{}的边际效果：** 在前序配置基础上，最佳验证准确率变化{:+.2f}个百分点，测试准确率变化{:+.2f}个百分点。该效果依赖此前累计设置，不能解释为对所有CNN的独立平均效果。".format(i-1,title,(r["validation_accuracy"]-previous["validation_accuracy"])*100,(r["test_accuracy"]-previous["test_accuracy"])*100))
    gap0=(rows[6]["clean_train_accuracy"]-rows[6]["test_accuracy"])*100
    gap1=(rows[7]["clean_train_accuracy"]-rows[7]["test_accuracy"])*100
    lines.append("8. **增强的泛化表现：** 增强前后训练—测试差距由{:.2f}变为{:.2f}个百分点；这里的训练集准确率均使用无增强数据、eval模式和验证选定的检查点，避免把增强后的在线训练准确率与干净测试准确率直接比较。".format(gap0,gap1))
    lines.append("9. **日志修正不是模型改进。** 最后两阶段只改变loss的显示口径，CSV都保存样本平均loss与batch均值之和。最终权重逐元素相同的检查结果记录在上方；这证明该统计修正不改变参数更新。")
    lines.append("10. **最终累计变化：** 测试准确率从{:.2%}到{:.2%}，变化{:+.2f}个百分点；参数量从{}到{}。大部分步骤未增加参数，观察到的变化不能简单归因为网络变大。".format(rows[0]["test_accuracy"],rows[-1]["test_accuracy"],(rows[-1]["test_accuracy"]-rows[0]["test_accuracy"])*100,rows[0]["parameters"],rows[-1]["parameters"]))
    selected=max(rows,key=lambda r:r["validation_accuracy"])
    lines.append("\n按验证集而非测试集选择的最高验证方案为`{}`，验证准确率{:.2%}，对应测试准确率{:.2%}；最后累计方案不必然是最佳方案。".format(selected["stage"],selected["validation_accuracy"],selected["test_accuracy"]))
    lines += ["\n## CNN结构层面的解释\n",
              "该模型通过卷积的局部连接与权重共享提取空间特征，三次池化将32×32降至4×4，再由分类头读取。卷积参数不随输入像素位置重复增加；ReLU不增加可学习参数。原模型仍有MaxPool非线性，不能称为纯线性网络，但两个连续Linear可合并为一个仿射变换。加入ReLU增加了特征变换能力，能否发挥作用受训练设置影响。",
              "本实验没有对卷积、权重共享或池化进行单独消融，所以上述是结构解释，而不是本实验对这些机制优越性的直接证明。也没有运行ViT，不能据此得出CNN优于ViT。",
              "\n## 边界、混杂因素与后续验证\n",
              "- 单个随机种子42，无法估计训练随机性的方差，也不能宣称统计显著。建议以同样划分复跑种子0、1、2等，报告均值与标准差。",
              "- 这是预先固定的累加对照，不是完整因子实验；后一步的效应依赖前一步设置。数据增强同时包含裁剪和翻转，不能分离二者贡献。",
              "- BatchNorm阶段同时移除了卷积bias，参数量略变；应称为BatchNorm配置的组合效果。若要严格分离BN作用，需要保留bias的额外对照。",
              "- 所有方案训练完后才查看官方测试集，各阶段检查点只用验证集选择。后续若根据这些测试结果修改方案，需要新的独立验证流程，不能反复利用测试集调参。",
              "- 当前45,000训练样本、固定种子、确定性设置与上一次50,000训练样本的随机基线不同，不能将旧的56.25%当成本轮严格配对的基线。",
              "- 运行耗时使用CPU张量缓存、固定线程数4及确定性CUDA，包含训练、验证和保存，不含最终统一评估。数据增强增加CPU工作，耗时不能只解释为网络推理成本；缓存等价性抽样检查见cache_equivalence.json。",
              "- 耗时来自单次顺序运行，仍受缓存预热、温控和系统负载影响，不属于重复测量的硬件基准。",
              "- 正式架构对照建议补充无ReLU的30轮实验、多个种子、不同学习率，以及保留卷积bias的BN对照。",
              "\n## 对后续CNN与ViT比较的启示\n",
              "本次同一小型CNN在不同训练配置下的结果可能差别很大，因此不应拿一个尚未充分优化的CNN与另一套训练配置下的ViT比较后，将全部差异归因于架构。后续应固定训练/验证/测试划分、统一数据处理和指标口径，并明确训练预算、优化器与调参预算；允许各架构使用适合自身的优化器时，也应清楚记录差异。本次没有ViT实测数据。",
              "\n## 各类识别表现\n", "|类别|基线召回率|最终召回率|变化(百分点)|", "|---|---:|---:|---:|"]
    for i,label in enumerate(CLASSES):
        lines.append("|{}|{:.2%}|{:.2%}|{:+.2f}|".format(label,recalls[0][i],recalls[1][i],(recalls[1][i]-recalls[0][i])*100))
    hardest_before = min(range(10), key=lambda i: recalls[0][i])
    hardest_after = min(range(10), key=lambda i: recalls[1][i])
    declined = ["{}（{:+.2f}个百分点）".format(CLASSES[i], (recalls[1][i]-recalls[0][i])*100)
                for i in range(10) if recalls[1][i] < recalls[0][i]]
    lines.append("\n**类别层面的发现：** 最低类别召回率从{}的{:.2%}提升到{}的{:.2%}。整体准确率提升并不保证每一类都改善；本次下降的类别为{}。各类训练与测试样本数相同，因此不能只用类别样本数量不平衡解释基线的类别表现差异；具体混淆原因仍需要查看误分类图像。".format(
        CLASSES[hardest_before], recalls[0][hardest_before], CLASSES[hardest_after], recalls[1][hardest_after],
        "、".join(declined) if declined else "无"))
    for name,row in [("基线",rows[0]),("最终",rows[-1])]:
        low,high=wilson(row["test_accuracy"])
        lines.append("\n{}测试准确率的Wilson 95%区间（仅条件于这一次训练和测试样本抽样，不含训练种子方差）：[{:.2%}, {:.2%}]。".format(name,low,high))
    lines += ["\n## 每步代码与动机\n",
              "以下为实际使用的关键改动，完整版本保存在各阶段目录中。所有阶段均从相同seed重新初始化，而不是在上一阶段权重上继续训练。",
              "1. ReLU：在每个Conv2d与MaxPool2d之间、两个Linear之间插入`nn.ReLU()`，增加非线性表达能力，不增加参数。",
              "2. 训练预算：`epoch = 10`改为`epoch = 30`，观察更充分优化的作用。",
              "3. 动量：`torch.optim.SGD(..., lr=0.01, momentum=0.9)`，综合历史更新方向；不是更换为另一类网络。",
              "4. 学习率：`StepLR(optimizer, step_size=10, gamma=0.1)`，每轮末尾执行`scheduler.step()`；第1—10轮lr=0.01，第11—20轮0.001，第21—30轮0.0001。",
              "5. 打乱：训练DataLoader使用`shuffle=True`，验证与测试保持False。",
              "6. 标准化：训练、验证、测试都在ToTensor后使用`Normalize((0.5,)*3, (0.5,)*3)`。这不是从测试集估计统计量，而是预先选定的固定映射。",
              "7. 增强：只在训练预处理开头添加`RandomCrop(32, padding=4)`与`RandomHorizontalFlip()`，让同一图片在不同轮次呈现位置和方向变化。",
              "8. BatchNorm：每个卷积后添加`nn.BatchNorm2d(out_channels)`再接ReLU，并将该卷积设为`bias=False`。BN增加的缩放/偏移参数与移除的bias需一并记录。",
              "9. loss口径：先累计`loss.item() * imgs.size(0)`，再除以数据集大小；这一统计量位于反向传播以外，不参与参数更新。",
              "\n## 文件与复现\n",
              "- `CNN_experiment_runner.py`：按预定次序生成脚本、逐步训练、统一测试。",
              "- `protocol.json`：预先固定的设置；`summary.csv/json`：总表；`changes.patch`：相邻版本逐行差异。",
              "- 每步目录的`CNNtrain_gpu1.py`保存当时版本，`metrics.csv`为逐轮结果，`best.pth/last.pth`为权重，`evaluation.json`包含混淆矩阵。",
              "- `original_CNNtrain_gpu1.py`为原始备份。当前主脚本保留最后一步设置。",
              "- `experiment_comparison.png/pdf`：学习曲线、阶段准确率、泛化差距和耗时；`class_recall.png`：类别召回率。",
              "- 图表编号0至9对应总表阶段顺序。以下为直接嵌入的结果图。",
              "\n![逐步实验结果](experiments/"+output.name+"/experiment_comparison.png)",
              "\n![类别召回率](experiments/"+output.name+"/class_recall.png)"]
    # 表格各行之间只能换行，段落之间才插入空行。
    chunks = []
    for i, block in enumerate(lines):
        if i:
            chunks.append("\n" if block.startswith("|") and lines[i-1].startswith("|") else "\n\n")
        chunks.append(block)
    report="".join(chunks)+"\n"
    local_report = report.replace("experiments/"+output.name+"/", "./")
    (output/"analysis.md").write_text(local_report,encoding="utf-8")
    log=output.parent.parent/("CNN实验日志_"+output.name+".md")
    prefix = log.read_text(encoding="utf-8").split("\n\n# 实测汇总与分析")[0]
    log.write_text(prefix + report, encoding="utf-8")
    print("Analysis and figures written:",log)


if __name__=="__main__": main()
