# 工单三：双招股说明书 RAG 问答

一个简单可运行的双PDF RAG项目：MinerU API解析《招股说明书1.pdf》和《招股说明书2.pdf》，本地BGE + FAISS + 字符关键词混合检索，DeepSeek API生成答案。

系统会根据问题中的公司名称自动选择对应PDF：兴图新科检索PDF1，力源信息检索PDF2。网页中的“批量评测”会运行14个指定问题，显示RAG、纯LLM、参考答案及关键事实准确率。

## 1. 填写 API 密钥

打开项目根目录下的 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
MINERU_API_KEY=你的MinerU密钥
```

## 2. 一次性解析和建库

在 PowerShell 中运行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单3"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU是异步解析，首次会依次处理两份PDF，需要等待。每份解析结果会独立缓存，之后重新建库不会重复解析已有结果。

## 3. 启动网页

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单3"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

浏览器打开：<http://127.0.0.1:8000>

## 文件说明

- `prepare.py`：调用MinerU解析两份PDF，分块并建立统一混合索引。
- `rag.py`：公司识别、PDF路由、混合检索、缓存、超时降级和DeepSeek生成。
- `app.py`：问答、14题自动评分、分PDF统计、纯LLM对比和反馈接口。
- `index.html`：用户界面。
- `questions.json`：PDF1的10题和PDF2的4题，共14题。
- `test_workorder3.py`：公司路由、双PDF元数据和问题列表测试。

> 单次问答在 2.75 秒处主动停止等待 DeepSeek，并降级展示检索原文，从而尽量把总耗时控制在 3 秒内。API 网络状况仍可能影响实际时间，网页会显示每次实测耗时。
