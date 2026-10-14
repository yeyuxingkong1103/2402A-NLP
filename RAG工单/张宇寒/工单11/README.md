# 工单11：金融领域Embedding微调

本项目使用金融年报语料训练一个轻量领域Embedding适配器，完整实现任务要求中的六个阶段：

1. 数据集生成；
2. 问答对及查询/正例/难负例三元组生成；
3. 数据集与基础Embedding模型加载；
4. Triplet Loss、训练参数与适配器训练；
5. Recall@K、MRR、相似度间隔评估器；
6. 微调前后检索效果对比。

## 使用环境

项目使用本机已有环境，不需要重新安装或修改环境：

```text
C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe
```

## 最简单的运行方法

直接双击 `启动项目.bat`，然后访问：

```text
http://127.0.0.1:8022
```

现成的训练数据、模型和评估报告已经随项目提供，因此第一次打开不需要重新训练。

## 从头执行六个步骤

双击 `重新训练.bat`，或在该环境终端中依次执行：

```powershell
python prepare.py
python train.py --epochs 60 --dimension 96
python app.py
```

## 为什么采用轻量适配器

任务资料中的`BAAI/bge-base-en-v1.5`是英文基础模型示例。本机缓存中存在中文`BAAI/bge-small-zh-v1.5`，但当前PyTorch/Transformers组合在导入时存在兼容性问题。为满足“不改变当前环境”和“代码简单可实现”的要求，默认方案使用TF-IDF字符嵌入与SVD作为冻结基础模型，只训练96维线性投影适配器。它仍然是真实的Embedding微调：Triplet Loss直接调整向量空间，模型和微调前后指标都可以保存、加载和复现。

## 主要文件

- `prepare.py`：生成问答对和三元组数据集；
- `dataset.py`：问题解析、正例匹配、难负例选择；
- `embedding_model.py`：基础Embedding和可训练适配器；
- `train.py`：训练参数、损失训练、模型保存；
- `evaluator.py`：微调前后检索评估；
- `app.py`、`index.html`：网页展示与检索体验；
- `data/model/evaluation_report.json`：完整评估结果。

