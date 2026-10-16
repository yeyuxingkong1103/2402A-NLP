# 工单11 研发说明（README）

**工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务**

## 1. 目录结构

```
研发/
├── src/                     全部源码（11 个模块，每个文件头部含工单编号注释）
│   ├── config.py            路径与超参统一加载（部署/配置/*.json + 环境变量覆盖）
│   ├── text_utils.py        归一化 / 分句 / Jaccard / 文档名解析 / 关键词命中
│   ├── generate_dataset.py  ① 数据集生成（三种格式 + 页级留出 + 统计与 MD5）
│   ├── losses.py            ③ 损失函数：Triplet / CosineSimilarity / Contrastive / Matryoshka
│   ├── encoder.py           ② 模型加载（本地离线 + CLS 池化 + L2 归一化 + 冻结策略 + OOM 重试）
│   ├── train.py             ④ 训练循环（AdamW + 预热/余弦退火 + 日志 + 曲线 + 检查点）
│   ├── evaluator.py         ⑤ 指标实现（Recall@K / MRR / nDCG@K / 相似度间隔 / 自助法 CI）
│   ├── evaluate.py          ⑥ 微调前后对照（留出测试集 + 真实索引端到端 + 对比表）
│   ├── plot.py              PIL 折线图 / 柱状图（本机无 matplotlib）
│   ├── benchmark_cpu.py     CPU 吞吐标定（训练规模决策依据）
│   └── inspect_corpus.py    语料勘察（分块/类型/字符数/指标命中率）
├── data/
│   ├── dataset/             train/dev/test.jsonl、留出测试集.jsonl、数据集统计.json
│   ├── finetuned/           微调后模型（512 维）+ 训练摘要.json + 微调信息
│   ├── train_log.jsonl      逐步训练日志
│   ├── loss_curve.png       损失曲线
│   └── 性能标定.json        CPU 吞吐实测
├── requirements.txt
└── README.md
```

## 2. 运行方式

```powershell
cd E:\gao6gongdan\工单11
$env:PYTHONDONTWRITEBYTECODE='1'

# ① 数据集生成（约 40 秒）
pwsh -NoProfile -File run_py.ps1 研发/src/generate_dataset.py

# ② 微调训练（CPU，约 3 分钟 / 117 步 = 1 个 epoch）
pwsh -NoProfile -File run_py.ps1 研发/src/train.py

# ③ 微调前后评估（约 3 分钟，检索池抽样 2500 块）
pwsh -NoProfile -File run_py.ps1 研发/src/evaluate.py

# ④ 一键流水线
pwsh -NoProfile -File 部署/脚本/一键流水线.ps1
```

## 3. 关键实现说明

1. **为什么不用 sentence-transformers 的 Trainer**：本机无 `accelerate`，Trainer 直接 ImportError；且工单原文点名的损失函数只有十几行，自实现更可控。训练循环在 `train.py`，一次性前向 anchor/positive/negative 三个向量，同时供 TripletLoss 与 CosineSimilarityLoss 使用。
2. **池化必须用 CLS**：底座 `1_Pooling/config.json` 标明 `pooling_mode_cls_token=true`，改成 mean pooling 会让微调前后向量空间不可比。
3. **防泄漏**：数据集按「页」留出 12%，留出页的分块既不做正例也不做负例；工单7 的 17 道真题全程只用于最终评估。
4. **降采样口径**：CPU 吞吐实测 8.77 序列/秒（seq128），故训练三元组截取 1400 条（种子 20260826 可复现），117 步 = 恰好 1 个完整 epoch；检索池抽样 2500 块且包含全部金标准分块。
5. **容错**：编码 OOM 自动折半批大小；训练异常或超预算均落盘检查点与摘要；语料损坏行跳过并计数。

## 4. 依赖

见 `requirements.txt`。**本工单未安装任何新依赖**（禁止裸 pip install，落点会写进只读的工单1）。
