# 招股说明书 RAG 问答

一个简单的本地 RAG 项目：MinerU API 解析 PDF，本地 BGE + FAISS 检索，DeepSeek API 生成答案。

## 1. 填写 API 密钥

打开项目根目录下的 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
MINERU_API_KEY=你的MinerU密钥
```

## 2. 一次性解析和建库

在 PowerShell 中运行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU 是异步解析，首次建库需要等待。上传、下载遇到临时断线会自动重试，下载支持续传，任务编号会保存在 `data/mineru/task.json`。建好后会复用 `data` 目录，不会每次重新解析；FAISS 索引读写兼容中文项目路径。

## 3. 启动网页

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

浏览器打开：<http://127.0.0.1:8000>

## 文件说明

- `prepare.py`：调用 MinerU、切分文本、建立 FAISS 索引。
- `rag.py`：Query 理解、本地检索和 DeepSeek 生成。
- `app.py`：问答、批量评测和反馈接口。
- `index.html`：用户界面。
- `questions.json`：10 个固定评测问题。

> 3 秒是优化目标。本地检索通常很快，总耗时主要受 DeepSeek API 和网络影响；网页会显示实际耗时。
