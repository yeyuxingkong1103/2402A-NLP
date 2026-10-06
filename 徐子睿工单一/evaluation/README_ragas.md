# RAGAS 标准评估（可选依赖）

> 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
> 模块：`run_ragas.py`

本目录的其余脚本只依赖主项目环境（FastAPI / numpy / jieba / PyMuPDF + Ollama）。
RAGAS 会引入 langchain 一整套依赖，**容易与主环境版本冲突**，因此单独建一个 venv，不污染主项目。

## 一、准备独立环境

```bat
cd /d D:\桌面\徐子睿工单一
python -m venv .venv_ragas
REM 用 uv 装（快且能把版本解干净）；没有 uv 就用 pip
uv pip install --python .venv_ragas\Scripts\python.exe ^
  "ragas==0.2.15" "langchain-community<0.4" "langchain<0.4" "langchain-openai<0.4"
```

> 版本坑：`ragas>=0.3` 会 import `langchain_community.chat_models.vertexai`，
> 而 `langchain-community>=0.4` 已删除该模块 → 必须把 langchain 系降到 `0.3.x`。
> `ragas==0.2.15` + `langchain-community 0.3.31 / langchain-core 0.3.86 / langchain-openai 0.3.35` 实测可用。

## 二、准备评测数据

RAGAS 需要「问题 / 答案 / 检索上下文 / 参考答案」四元组。先跑一次 RAG 链路把答案和上下文落盘：

```bat
python evaluation\build_bilingual_qa.py     REM 产出 evaluation\qa_bilingual_result.json
```

## 三、跑 RAGAS

```bat
.venv_ragas\Scripts\python.exe evaluation\run_ragas.py --lang zh
.venv_ragas\Scripts\python.exe evaluation\run_ragas.py --lang en
```

- 评判 LLM / 嵌入模型走本机 Ollama 的 OpenAI 兼容端点（`http://localhost:11434/v1`，`qwen2:7b` + `bge-m3`），
  **与系统同源、不调用外部 API、不产生费用**；可用 `--gen` / `--embed` / `--base` 换成更强的裁判模型。
- 指标：`faithfulness`、`answer_relevancy`、`llm_context_precision_without_reference`、
  `context_recall`、`answer_correctness`（后两者需要参考答案）。
- 结果落盘：`ragas_result_zh.json` / `ragas_result_en.json`（含总分与逐题明细）。
- 小模型裁判偶尔给不出合法 JSON，脚本用 `raise_exceptions=False`，无法解析的样本计为 NaN 并在
  `*_n` 字段记录有效样本数，不会中断整体评估。

## 四、耗时参考

本机（RTX 4060 + Ollama qwen2:7b，串行）约 **7 s / 指标·题**：10 题 × 5 指标 ≈ **6 分钟 / 语言**。
