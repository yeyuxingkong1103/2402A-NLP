# 工单六：招股说明书多策略检索问答

本项目在工单五双PDF多轮问答基础上增加三种可配置检索方式和三种重排方式，用现有17道问题比较优化前后的回答准确率、检索召回率与平均耗时。

## 功能

- 向量检索：本地BGE编码，FAISS余弦相似度召回。
- 全文检索：字符级TF-IDF匹配金额、年份、企业名称和专业术语。
- 混合检索：可在网页调整向量权重，全文权重自动为 `1-向量权重`。
- 重排：TF-IDF、DeepSeek LLM、用户反馈自适应，也可关闭重排作为基线。
- LLM重排失败或超时时自动降级为TF-IDF重排。
- 保留双PDF、会话记忆、引用页码、PDF查看、反馈和RAG/纯LLM对比。
- 一键用17题展示优化前后准确率、召回率和平均响应时间。

准确率90%和召回率95%是目标值，网页展示的是填写API并完成建库后的真实评测结果，代码不会伪造达标数据。

## 1. API配置

打开项目根目录的 `.env`，填写：

```env
DEEPSEEK_API_KEY=
MINERU_API_KEY=
```

## 2. 一次性解析和建库

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单6"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

MinerU上传、轮询和下载均支持重试与续传。已有解析缓存时可直接重新建库；FAISS索引兼容中文项目路径。

## 3. 启动网页

可以在PyCharm中右击运行 `app.py`，也可执行：

```powershell
cd "C:\Users\lenovo\Desktop\招股说明书RAG-工单6"
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" app.py
```

浏览器打开：<http://127.0.0.1:8017>

## 4. 使用方式

1. 选择向量、全文或混合检索。
2. 选择TF-IDF、LLM、反馈自适应或不重排。
3. 混合检索时调整向量权重，并选择返回引用数量。
4. 正常提问；回答下方显示实际策略、检索耗时、PDF页码和原文。
5. 点赞或点踩会影响后续“反馈自适应”排序。
6. 在“批量评测”中点击“比较优化前后检索”，查看17题真实指标。

## 主要文件

- `prepare.py`：MinerU解析、文本分块和FAISS/TF-IDF建库。
- `rag.py`：问题理解、三种检索、分数融合、三种重排和DeepSeek生成。
- `feedback.py`：本地反馈统计和自适应加减分。
- `conversation.py`：多轮上下文与跨公司追问。
- `app.py`：问答、反馈、评测及启动入口，端口8017。
- `index.html`：策略配置、对话和评测页面。
- `questions.json`：17道固定评测题。
- `test_workorder6.py`：工单6功能测试。

本项目沿用现有 `fastapi_FAQ_chat` 环境，不需要安装新组件。`.env`、`data/`、`feedback.json`、缓存和PDF不会上传Git。
