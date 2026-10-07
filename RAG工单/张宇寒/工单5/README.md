# 工单五：招股说明书多轮问答

一个简单可运行的双PDF RAG项目：MinerU API解析《招股说明书1.pdf》和《招股说明书2.pdf》，本地BGE + FAISS + 字符关键词混合检索，DeepSeek API生成答案。

在工单四的基础上，工单五增加会话记忆。可以连续问“他参与的哪个工程获奖？”“这个公司的法定代表人是谁？”“那武汉力源信息技术股份有限公司呢？”。网页保留每轮问题、答案、PDF原文依据和反馈。力源信息组织结构图的销售处问题已加入固定评测，现共17题。

会话记忆只在本机服务运行期间保存，点击“新对话”或重启服务后重新开始；不额外调用 LLM 改写问题，因此不会增加一次 API 请求。

## 1. 填写 API 密钥

打开项目根目录下的 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
MINERU_API_KEY=你的MinerU密钥
```

## 2. 一次性解析和建库

在 PowerShell 中运行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单5"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU是异步解析，首次会依次处理两份PDF，需要等待。上传、下载遇到临时断线会自动重试，下载支持续传，每份文档的任务编号会保存在对应缓存目录。解析结果会独立缓存，之后重新建库不会重复解析；FAISS索引读写兼容中文项目路径。

## 3. 启动网页

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单5"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8016
```

浏览器打开：<http://127.0.0.1:8016>

## 文件说明

- `prepare.py`：调用MinerU解析两份PDF，并把两张已核对图表加入统一混合索引。
- `figure_facts.json`：原PDF两张图的文字、图题、公司和页码。
- `rag.py`：公司识别、PDF路由、混合检索、缓存、超时降级和DeepSeek生成。
- `conversation.py`：会话记忆、指代消解和跨公司追问。
- `app.py`：多轮问答、17题自动评分、原PDF查看、纯LLM对比和反馈接口。
- `index.html`：聊天界面。
- `questions.json`：PDF1的10题和PDF2的7题，共17题。
- `test_workorder5.py`：多轮追问、公司路由、图表索引、问题列表测试。

> 图表文字仍针对已核对的两张图，不是通用图片OCR。单次问答在2.75秒处主动停止等待DeepSeek并展示检索原文；实际时间仍取决于本地检索和网络，网页会显示实测值。90%准确率和3秒响应是目标，须在填写密钥并完成建库后用批量评测验证，不能仅凭代码保证。未填写密钥、未运行建库前只能查看界面。
