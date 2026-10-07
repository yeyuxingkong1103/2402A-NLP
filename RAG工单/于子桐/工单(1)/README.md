# 工单01 —— 基于 PDF 文档的问答系统

工单编号: **人工智能NLP-RAG-基于PDF文档的问答系统**

对《招股说明书1.pdf》(武汉兴图新科电子股份有限公司) 建立向量索引, 提供检索增强问答 (RAG)
与评估能力, 并提供 Streamlit 交互界面。

---

## 一、功能一览

| 脚本 | 功能 |
|---|---|
| `build_index.py` | 解析 PDF -> 文本分块 (500 字 / 50 重叠) -> bge 嵌入 -> 落盘索引 |
| `qa_eval.py` | 10 个测试问题的 RAG 问答 + 纯 LLM 基线对比 + RAGAS 风格评估, 生成结果报告 |
| `app.py` | Streamlit 问答界面 (检索方式 / top_k / 重排可调, 展示命中片段) |
| `rag_common.py` | 共享内核: PDF 解析、分块、嵌入、向量/全文/混合检索、重排、LLM、评估 |

技术方案 (工单01 基线): **pypdf 纯文本抽取 + 500/50 分块 + 向量检索 top5 + DeepSeek 生成**。

---

## 二、环境依赖

- Python: 本机使用 `C:\Users\Lenovo\Desktop\ai\python.exe` (Python 3.11+)
- pip 包:

  ```bash
  pip install pypdf pdfplumber sentence-transformers jieba numpy scikit-learn openai streamlit
  ```

- 本地模型 (CPU 推理):
  - 嵌入模型: `D:\专高三资料\bge-base-zh-v1.5` (默认, 可用环境变量 `RAG_EMBED_MODEL` 覆盖)
  - 重排模型: `D:\专高三资料\bge-reranker-base` (可选, 界面勾选重排时使用)
- LLM: DeepSeek API, 在 `.env` 中配置:

  ```
  LLM_API_KEY=sk-xxxxxxxx
  ```

  (代码不硬编码 key; `rag_common.py` 依次从环境变量 `DEEPSEEK_API_KEY`、本目录 `.env` 读取)

---

## 三、目录结构

```
工单(1)/
├── 附件/
│   └── 招股说明书1.pdf        # 原始文档 (488 页)
├── build_index.py             # 建索引脚本
├── index/
│   └── doc1_text/             # 索引 (1299 块)
│       ├── chunks.jsonl       # 分块文本 (含页码/来源)
│       ├── emb.npy            # 归一化向量矩阵
│       └── meta.json          # 块数 / 维度
├── qa_eval.py                 # 评估脚本
├── output/
│   ├── qa_results.json        # 逐题明细 (答案/检索片段/各项指标)
│   └── 结果报告.md            # 评估报告 (可直接提交)
├── app.py                     # Streamlit 问答界面
├── rag_common.py              # 共享内核 (与其余工单同一份)
├── .env                       # LLM_API_KEY (不提交)
└── README.md
```

问题与参考答案不在本目录硬编码, 统一读取 `../_build/`:

- `../_build/questions.json` 的 `doc1` 段: 10 个测试问题
- `../_build/references.json` 的 `doc1` / `doc1_keywords` 段: 人工核对过的参考答案与关键词

---

## 四、使用方法

```bash
cd "C:/Users/Lenovo/Desktop/于子桐/工单(1)"

# 1. 构建索引 (已建好, 存在时自动跳过; 重建请先删除 index/doc1_text)
python build_index.py

# 2. 运行评估 (约 3-5 分钟, 需调用 DeepSeek API)
python -u qa_eval.py > qa_eval.log 2>&1

# 3. 启动问答界面
streamlit run app.py
```

`streamlit run` 启动后浏览器打开 http://localhost:8501 (端口被占用可加
`--server.port 8502`)。

界面说明:

- 侧边栏: 检索方式 (向量/全文/混合)、top_k (1-10)、重排方式 (不重排 / TF-IDF / Cross-Encoder)
- 主区: 底部输入框提问, 回答按对话流展示
- 每条回答下方「命中片段」展开区: 页码 + 相似度分数 + 原文片段

---

## 五、评估结果 (2026-10-06 重跑, 参考答案取自 ../_build/references.json)

索引 1299 块, 向量检索 top_k=5, 生成模型 DeepSeek-Chat。

| 指标 | RAG (向量检索) | 纯 LLM 基线 (无检索) |
|---|---|---|
| 答案正确性 | **0.700** | 0.420 |
| 忠实度 | 0.900 | - |
| 答案相关性 | 0.990 | - |
| 上下文精度 | 0.420 | - |
| 上下文召回 | 0.830 | - |
| 关键词命中率 (客观指标) | **0.683** | 0.518 |

- 平均响应: 总 3.23s = 检索 1.61s + 生成 1.62s (生成耗时受 DeepSeek 网络往返影响)
- RAG 相对纯 LLM 答案正确性提升 **+0.280**; 数字类问题 (注册资本 5,520 万元、
  募资补流 15,000 万元等) RAG 全部答对, 纯 LLM 全部答错
- 存在的短板: 表格内数字 (如军用领域收入明细、收入占比) 以纯文本解析会错行,
  相关题目 (260 / 33) 检索不到正确数字 —— 表格解析优化见工单02/03

逐题答案、参考答案与检索片段见 `output/结果报告.md`; 完整明细见 `output/qa_results.json`
(工单02 的前后对比评估会直接复用该文件作为"优化前"基线)。

---

## 六、常见问题

| 现象 | 处理 |
|---|---|
| `索引不存在` 报错 | 先运行 `python build_index.py` |
| `未找到 API Key` | 设置环境变量 `DEEPSEEK_API_KEY`, 或在本目录 `.env` 中配置 `LLM_API_KEY=sk-...` |
| 界面首问很慢 | 首次提问要加载嵌入模型 (约 10s), 之后有缓存 |
| 8501 端口被占用 | `streamlit run app.py --server.port 8502` |
| 中文控制台乱码 | 脚本已加 UTF-8 stdout 重定向; cmd 里可先执行 `chcp 65001` |
