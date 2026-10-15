# 工单11：Embedding模型微调

按设计、研发、测试、优化、部署五部分保存。

策略hybrid：Hit@5=100.0%，MRR@5=0.875，事实关键词覆盖代理=100.0%。
真实生成18题，完整稳态耗时710.00-6803.43ms；当前样本3秒目标未通过。不包含模型冷启动。

一键启动：双击一键启动.cmd，打开http://127.0.0.1:8775 。

所有源PDF、嵌入模型、索引、图谱、图像、测试集与日志均位于本工单目录。本项目运行时不访问其他工单文件。LLM调用本机已有Ollama服务，其环境与模型名称单独记录；主目录无公共数据。

各项是否满足验收阈值以测试/实测结果与专项记录为准。未通过不冒称通过。

专项结果 微调实测结果.json
{
  "base_model": "BAAI/bge-small-zh-v1.5",
  "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "loss": "MultipleNegativesRankingLoss / in-batch InfoNCE",
  "lr": 1e-05,
  "batch_size": 4,
  "epochs": 5,
  "max_seq_length": 256,
  "splits": {
    "train": 23,
    "validation": 5,
    "test": 8
  },
  "page_group_leakage": false,
  "selection": "仅按验证集MRR选择checkpoint，测试集只做前后评估",
  "before": {
    "sample_count": 8,
    "recall_at_1": 0.75,
    "recall_at_5": 0.875,
    "mrr": 0.8238636363636364,
    "ranks": [
      11,
      1,
      1,
      1,
      2,
      1,
      1,
      1
    ],
    "candidate_count": 445
  },
  "after": {
    "sample_count": 8,
    "recall_at_1": 0.875,
    "recall_at_5": 0.875,
    "mrr": 0.8828125,
    "ranks": [
      16,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    "candidate_count": 445
  },
  "training": [
    {
      "epoch": 1,
      "mean_loss": 0.32412972549597424,
      "validation": {
        "sample_count": 5,
        "recall_at_1": 0.4,
        "recall_at_5": 0.8,
        "mrr": 0.6004694835680751,
        "ranks": [
          1,
          2,
          1,
          426,
          2
        ],
        "candidate_count": 445
      }
    },
    {
      "epoch": 2,
      "mean_loss": 0.16154867317527533,
      "validation": {
        "sample_count": 5,
        "recall_at_1": 0.6,
        "recall_at_5": 0.8,
        "mrr": 0.7004950495049505,
        "ranks": [
          1,
          2,
          1,
          404,
          1
        ],
        "candidate_count": 445
      }
    },
    {
      "epoch": 3,
      "mean_loss": 0.051535147067625076,
      "validation": {
        "sample_count": 5,
        "recall_at_1": 0.6,
        "recall_at_5": 0.8,
        "mrr": 0.6505221932114883,
        "ranks": [
          1,
          4,
          1,
          383,
          1
        ],
        "candidate_count": 445
      }
    },
    {
      "epoch": 4,
      "mean_loss": 0.033861391770187765,
      "validation": {
        "sample_count": 5,
        "recall_at_1": 0.6,
        "recall_at_5": 0.8,
        "mrr": 0.6505361930294906,
        "ranks": [
          1,
          4,
          1,
          373,
          1
        ],
        "candidate_count": 445
      }
    },
    {
      "epoch": 5,
      "mean_loss": 0.01270869211293757,
      "validation": {
        "sample_count": 5,
        "recall_at_1": 0.6,
        "recall_at_5": 0.8,
        "mrr": 0.6505390835579515,
        "ranks": [
          1,
          4,
          1,
          371,
          1
        ],
        "candidate_count": 445
      }
    }
  ],
  "improved": true,
  "limits": "小规模合成问答实验，不保证泛化；工单验收的“变好”只在实测指标提升时成立"
}

## 验收范围说明
功能、材料和短时回归已交付；仍有部分指定问题完整耗时超过3秒。检索命中和关键词指标不是90%问答准确率或95%全事实召回率的证明。生产高并发、24小时稳定性及真人语音未全部验证；全部原始结果见本项目测试目录。
