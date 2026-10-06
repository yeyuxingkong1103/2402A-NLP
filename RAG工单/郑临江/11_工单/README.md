# 工单 11：Embedding 模型微调

**工单编号**：人工智能NLP-RAG项目-Embedding模型微调任务

## 一、项目简介

在专业金融数据上微调 `BAAI/bge-base-en-v1.5` 嵌入模型，缩小“语义鸿沟”，
提升 RAG 系统检索准确性。实现数据集生成、问答对生成、数据集与模型加载、
损失函数定义、训练参数定义、评估器创建、微调前/后评估。

## 二、微调流程

```
data_gen.py（数据集生成）→ train.py（加载/损失/训练/评估）→ evaluate.py（前后对比）
```

## 三、损失函数

| 损失函数 | 适用数据 | 说明 |
|----------|----------|------|
| Triplet Loss | (锚点, 正例, 负例) | 拉近正例、推远负例 |
| Contrastive Loss | 正负例句子对 | 相似接近、相异远离 |
| Cosine Similarity Loss | 带相似度分数句对 | 余弦相似度对齐分数 |
| Matryoshka Loss | 截断嵌入 | 生成分层可截断嵌入 |

## 四、目录结构

```
11_工单/
├── data_gen.py     # 数据集生成（问答对）
├── train.py        # 微调主程序
├── evaluate.py     # 微调前后检索效果对比
├── config.py       # 配置（基础模型、损失、训练参数）
├── data/           # 生成的数据集
├── output_model/   # 微调后的模型
├── requirements.txt
└── README.md
```

## 五、运行

```bash
pip install -r requirements.txt   # 需 GPU 或 CPU（较慢）
python data_gen.py                # 生成数据集
python train.py                   # 微调 + 微调前/后评估
python evaluate.py                # 独立对比前后检索效果
```

## 六、验收对照

- 实现数据集生成、问答对生成、加载、损失函数、训练参数、评估器；
- 输出微调前后评估结果，微调后检索效果优于微调前（有数据指标支撑）；
- 代码注释含工单编号：人工智能NLP-RAG项目-Embedding模型微调任务。
