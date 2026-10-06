# RAG 基于 PDF 文档的问答系统

基于本地 **LangChain 2 + Ollama** 的检索增强生成（RAG）问答系统，以《招股说明书1.pdf》为知识源，可对文档内容提问并给出来源页码，内置 RAGAS 评估与「RAG vs 纯 LLM」对比。

**工单编号**：人工智能 NLP-RAG-基于 PDF 文档的问答系统

## 特性

- 🔍 **查询理解**：意图识别 / 指代消歧 / 子问题分解
- 📚 **混合检索**：向量检索（bge-m3 + FAISS）⊕ BM25 词法检索 → RRF 融合
- 🎯 **精准生成**：强约束提示词，严格依据文档回答 + 来源页码
- ⏱️ **快速响应**：生成长度控制 + 知识库缓存，目标 ≤ 3s
- ⚖️ **RAG vs 纯 LLM**：同一提问双链路对比，RAGAS 五项指标 + 人工命中检查
- 🖥️ **Streamlit 界面**：文字 + 语音 + 满意度反馈

## 快速开始

```powershell
# 1. 前置：已启动 Ollama，已拉取 deepseek-r1:1.5b 与 bge-m3:567m
conda activate langchain2
pip install -r requirements.txt

# 2. 启动 Web 界面
streamlit run app.py

# 3. 命令行快速问答
python scripts/run_qa.py "公司法定代表人和注册资本是多少？"

# 4. （可选）重建向量库
python scripts/build_kb.py --force
```

## 目录结构

```
├── app.py                  # Streamlit 界面
├── src/                    # 核心模块（config/pdf_parser/knowledge_base/
│                           #   query_understanding/retriever/qa_engine）
├── scripts/                # build_kb / run_qa / evaluate / build_report
├── data/                   # 招股说明书1.pdf + questions.json
├── vector_db/              # FAISS 向量库
├── output/                 # 评估报告 + 用户反馈
├── 技术文档.md  ｜ 用户手册.md ｜ requirements.txt ｜ .env.example
```

## 评估结果

运行 `python scripts/evaluate.py` 后用 `python scripts/build_report.py` 生成，产物为 `output/evaluation_report.md`（RAGAS 指标 + 逐题 RAG/LLM 耗时与命中对照）。

详见 [技术文档](技术文档.md) 与 [用户手册](用户手册.md)。