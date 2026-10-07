# 工单四：招股说明书图文问答

一个简单可运行的双PDF RAG项目：MinerU API解析《招股说明书1.pdf》和《招股说明书2.pdf》，本地BGE + FAISS + 字符关键词混合检索，DeepSeek API生成答案。

系统会根据公司名称选择对应PDF。工单四新增了组织结构图和2008年中国IC市场应用结构与增长图的检索；这两张图的文字由原PDF第39页和第72页人工核对后存入 `figure_facts.json`，建库时与MinerU解析结果一起索引。网页批量评测共16题，并可从图表证据打开原PDF核对。

## 1. 填写 API 密钥

打开项目根目录下的 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
MINERU_API_KEY=你的MinerU密钥
```

## 2. 一次性解析和建库

在 PowerShell 中运行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单4"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU是异步解析，首次会依次处理两份PDF，需要等待。上传、下载遇到临时断线会自动重试，下载支持续传，每份文档的任务编号会保存在对应缓存目录。解析结果会独立缓存，之后重新建库不会重复解析；FAISS索引读写兼容中文项目路径。

## 3. 启动网页

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单4"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8014
```

浏览器打开：<http://127.0.0.1:8014>

## 文件说明

- `prepare.py`：调用MinerU解析两份PDF，并把两张已核对图表加入统一混合索引。
- `figure_facts.json`：原PDF两张图的文字、图题、公司和页码。
- `rag.py`：公司识别、PDF路由、混合检索、缓存、超时降级和DeepSeek生成。
- `app.py`：问答、16题自动评分、原PDF查看、纯LLM对比和反馈接口。
- `index.html`：用户界面。
- `questions.json`：PDF1的10题和PDF2的6题，共16题。
- `test_workorder4.py`：公司路由、图表索引、问题列表测试。

> 图表文字目前针对工单四指定的两张图，不是通用图片OCR。单次问答在2.75秒处主动停止等待DeepSeek并展示检索原文；实际时间仍取决于本地检索和网络，网页会显示实测值。未填写密钥、未运行建库前只能查看界面。
