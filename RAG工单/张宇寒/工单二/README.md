# 招股说明书 RAG 问答

工单2的简单可运行版本：MinerU API 解析 PDF，本地 BGE + FAISS + 字符关键词混合检索，DeepSeek API 生成答案。

本版主要优化了三处：同页短段落合并并保留重叠、语义与关键词混合召回、问题结果缓存和 3 秒超时降级。网页中的“批量评测”会自动运行指定的 10 个问题，显示 RAG、纯 LLM、参考答案及关键事实准确率。

## 1. 填写 API 密钥

打开项目根目录下的 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
MINERU_API_KEY=你的MinerU密钥
```

## 2. 一次性解析和建库

在 PowerShell 中运行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单2"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU 是异步解析，首次建库需要等待。建好后会复用 `data` 目录，不会每次重新解析。工单2必须重新运行一次本步骤，以生成新的混合检索索引。

## 3. 启动网页

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单2"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

浏览器打开：<http://127.0.0.1:8000>

## 文件说明

- `prepare.py`：调用 MinerU、优化分块并建立语义/关键词混合索引。
- `rag.py`：Query 理解、混合检索、缓存、超时降级和 DeepSeek 生成。
- `app.py`：问答、10 题自动评分、纯 LLM 对比和反馈接口。
- `index.html`：用户界面。
- `questions.json`：10 个固定评测问题。

> 单次问答在 2.75 秒处主动停止等待 DeepSeek，并降级展示检索原文，从而尽量把总耗时控制在 3 秒内。API 网络状况仍可能影响实际时间，网页会显示每次实测耗时。
