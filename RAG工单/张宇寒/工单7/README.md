# 工单7：CCF年报 RAG 测试与检索评估

这是一个简单的本地RAG评测项目。它读取附件 `ccf_competition` 中已经解析好的9份年报文本，用10道固定问题重复测试检索效果，并在网页展示每题的RAG答案、Top K证据、页码、评测分数和耗时。

## 功能

- 读取9份金融机构年报的结构化TXT，不重复调用MinerU。
- 使用本地BGE模型、FAISS向量检索和TF-IDF全文检索。
- 支持向量、全文、混合三种检索模式。
- 固定10道题，可重复比较不同模式和Top K参数。
- 指标包括 Hit@K、Recall@K、MRR、事实准确率、平均耗时和P95耗时。
- 配置DeepSeek后生成简洁答案；未配置时自动展示检索证据摘要，检索评测仍可运行。
- 支持下载完整JSON评测报告，并可打开原始PDF核对页码。

## 使用的本地环境

项目直接使用已有环境，不需要新建或修改环境：

```text
C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe
```

## 1. 填写API（可选）

打开项目根目录 `.env`：

```env
DEEPSEEK_API_KEY=
```

把DeepSeek API Key填在等号后面。留空也可以运行检索评测，页面会使用“证据摘要模式”。工单7不需要MinerU API。

## 2. 一次性建立索引

第一次使用时运行 `prepare.py`：

```powershell
cd "C:\Users\lenovo\Desktop\CCF年报RAG-工单7"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

程序默认读取：

```text
C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition
```

如果附件移动了，可临时设置 `CCF_DATA_DIR` 指向解压后的 `ccf_competition` 目录。

## 3. 启动网页

索引成功后，可在PyCharm中右击运行 `app.py`，也可以执行：

```powershell
cd "C:\Users\lenovo\Desktop\CCF年报RAG-工单7"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" app.py
```

浏览器打开：<http://127.0.0.1:8018/>

## 4. 运行评测

1. 选择向量、全文或混合检索。
2. 选择Top K数量。
3. 点击“一键运行固定10题”。
4. 查看汇总指标与每题的参考答案、RAG答案、证据和耗时。
5. 点击“下载JSON报告”保存完整结果。

## 主要文件

- `prepare.py`：读取结构化TXT并建立FAISS和TF-IDF索引。
- `corpus.py`：报告识别、JSONL容错解析和文本分块。
- `retrieval.py`：本地混合检索与DeepSeek答案生成。
- `evaluation.py`：10题执行和指标计算。
- `evaluation_questions.json`：固定10题、参考答案与目标证据。
- `app.py`：FastAPI接口和8018端口启动入口。
- `index.html`：评测网页。
- `tests/`：自动化测试。

`.env`、`data/`、评测结果、原始PDF和结构化TXT不会上传到Git。
