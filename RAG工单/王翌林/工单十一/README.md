# Embedding 模型微调任务 —— 工单十一

> 工单编号：人工智能NLP-RAG-Embedding模型微调任务

本项目以工单十为基线复制，位于 `/home/dabaie/code/工单/工单十一`。针对金融领域检索场景，用年报语料自动生成问答对数据集，对 **BAAI/bge-m3** 进行对比损失微调（MultipleNegativesRankingLoss），并按工单规定流程完成**微调前后同口径检索评估**。

## 验收结论（数据支撑）

**7/7 项检索指标全部提升，微调后优于微调前**（评估：54 查询 × 1016 语料 chunk，同口径复测）：

| 指标 | 微调前 | 微调后 | Δ |
| --- | --- | --- | --- |
| mrr@10 | 0.7055 | 0.7798 | +0.0743 |
| ndcg@10 | 0.7363 | 0.8015 | +0.0652 |
| map@100 | 0.6204 | 0.6989 | +0.0785 |
| accuracy@1 | 0.6296 | 0.6852 | +0.0556 |
| accuracy@3 | 0.7593 | 0.8519 | +0.0926 |
| accuracy@5 | 0.8148 | 0.9259 | +0.1111 |
| accuracy@10 | 0.8704 | 0.9630 | +0.0926 |

完整数据：[docs/finetune_v11_results.json](docs/finetune_v11_results.json)（含训练 loss 曲线日志）

## 工单十一新增内容

```
src/finetune_v11/qa_dataset.py   # 数据集生成：prompt构造/LLM输出解析/分层抽样/防泄漏切分
src/finetune_v11/ir_eval.py      # 评估器输入构造 + 指标抽取与前后对比
scripts/gen_qa_pairs_v11.py      # 运行问答对生成（DeepSeek，216 chunk→432 对，零失败）
scripts/finetune_bge_v11.py      # 微调主脚本：加载→损失→参数→评估→训练→同口径复测
tests/test_finetune_v11.py       # 14 个单元测试
data/finetune_v11/               # 生成的数据集（train 388 / dev 44）+ 评估语料 1016 chunks
models/bge-m3-ft-v11/            # 微调后模型（sentence-transformers 格式，2.2GB）
docs/00_工单十一任务说明.md       # 需求对照 / 数据集设计 / 验收对照
docs/12_微调实现步骤与问题记录.md # 实现步骤 + 训练过程 + 前后指标 + 4 个过程问题
docs/finetune_v11_results.json   # 训练过程与微调前后指标（验收数据）
docs/screenshots/                # 测试截图（终端实证 + 训练曲线 + 指标对比）
```

## 工单要求对照

| 类别 | 要求 | 落地 |
| --- | --- | --- |
| 设计 | 技术组件 | [docs/技术组件.md](docs/技术组件.md) |
| 设计 | 技术架构 / 流程图 | [docs/技术架构与流程图.md](docs/技术架构与流程图.md) |
| 设计 | 思维导图 | [docs/思维导图.md](docs/思维导图.md) |
| 设计 | 接口文档 | [docs/API接口文档.md](docs/API接口文档.md) |
| 研发 | 代码 | `src/finetune_v11/`、`scripts/finetune_bge_v11.py`、`scripts/gen_qa_pairs_v11.py` |
| 测试 | 截图（多截图） | [docs/screenshots/](docs/screenshots/) 共 8 张 |
| 部署 | linux shell 脚本 | `scripts/gen_qa_pairs_v11.py`、`scripts/finetune_bge_v11.py` |
| 部署 | 安装脚本（conda/cuda 环境） | `scripts/install_conda_env.sh`（conda env + PyTorch cu121） |
| 部署 | 启动脚本 | `scripts/finetune_bge_v11.py`（基线评估→训练→复测一体化） |
| 部署 | 结束脚本 | 训练自动结束并保存模型，无需手动停止 |

## 测试截图索引（docs/screenshots/）

| 文件 | 内容 |
| --- | --- |
| 01_finetune_metrics.png | 微调前后 7 项指标对比表（before/after/delta） |
| 02_train_loss_log.png | 训练 loss 日志（9 步记录，0.2506→0.0441） |
| 03_dataset.png | 生成的数据集文件列表（train 388 / dev 44） |
| 04_model_dir.png | 微调后模型目录（bge-m3-ft-v11，2.2GB） |
| 05_unit_tests.png | 14 个单元测试运行结果（全过） |
| 06_gen_report.png | 问答对生成报告 |
| 07_train_loss_curve.png | 训练 Loss 曲线 + 学习率衰减图 |
| 08_metrics_comparison.png | 微调前后指标柱状图对比（7/7 提升） |

## 快速复现

```bash
cd /home/dabaie/code/工单/工单十一
PY=/home/dabaie/code/my_project/.venv/bin/python

$PY scripts/gen_qa_pairs_v11.py     # 1) 生成问答对数据集（约3-5分钟，需 .env DEEPSEEK_API_KEY）
$PY scripts/finetune_bge_v11.py     # 2) 基线评估→训练→复测（GPU 约2分钟）
```

关键训练配置：冻结嵌入层+底部12层（可训练 152M/568M）、bf16、batch 8、lr 2e-5、epochs 2、max_seq 256、seed 42——8GB 显存可训，实测训练 19.4s、loss 0.2506→0.0441。

## 质量

- 新增单测 14 用例全过；全量 pytest 回归 **300 passed / 1 skipped**，无回归
- 新增代码注释均含工单编号：**人工智能NLP-RAG-Embedding模型微调任务**

---

# 附：基线系统（工单六/七/十摘要）

- **工单六（混合检索）**：向量（bge-m3+Milvus+重排）/全文（倒排+布尔/短语/模糊）/混合（RRF、加权平均）三策略可配；准确率 93.8%、召回 96.9%
- **工单七（功能测试及评估）**：9 份 A 股年报语料（7878 chunks）+ 10 题评估，答案准确率 10/10、doc_recall@5 0.96、MRR 0.95
- **工单十（Docker 部署）**：rag-api(8006)+rag-ui(8506) 容器化，named volume 持久化、自定义网络
- 本工单的微调模型 `models/bge-m3-ft-v11/` 可替换 v6 引擎中的 bge-m3 路径直接生效
