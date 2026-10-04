# 工单 07：功能测试及评估

**工单编号**：人工智能NLP-RAG-功能测试及评估

## 一、项目简介

基于工单 01~06 实现的 RAG，对 `ccf_competition.zip` 中的金融年报 PDF
进行 RAG 检索测试与评估。使用 `sample_questions.pdf` 中的示例问题整理出
10 个测试问题，通过 RAGAS 风格评估框架输出评估结果，并分析检索问题。

## 二、功能

1. 加载 ccf 年报（txt 优先，可回退 pdf）；
2. 对 10 个问题执行 BM25 检索；
3. 计算四项评估指标（忠实度、答案相关性、上下文精度、上下文召回）；
4. 输出 `results.json` 与结果分析（`results.md`）。

## 三、目录结构

```
07_工单/
├── test.py        # 测试主程序
├── evaluation.py  # RAGAS 风格评估框架
├── config.py      # 10 个问题与参考答案
├── retriever.py   # BM25 检索
├── results.md     # 结果与分析
├── requirements.txt
└── README.md
```

## 四、运行

```bash
pip install -r requirements.txt
python test.py   # 生成 results.json
```

## 五、验收对照

- 10 个问题体现 ccf 年报内容相关性；
- 使用 01~06 的 RAG 进行测试；
- 输出 10 个问题的检索结果 + 评估结果 + 问题分析；
- 代码注释含工单编号：人工智能NLP-RAG-功能测试及评估。
