# 工单8：金融年报 Graph RAG 问答系统

本项目直接读取附件 `ccf_competition.zip` 中9份金融机构年报的TXT内容，构建本地知识图谱，并根据 `eval_questions.md` 中10道固定问题完成检索问答。网页会同时展示答案、Top K文本块、年报页码和当前答案对应的知识图谱子图。

## 已实现功能

- 直接读取ZIP内9份年报TXT，不重复调用MinerU。
- 构建公司、年报、年份、金融指标、文本块五类图谱节点。
- 建立 `HAS_REPORT`、`REPORT_YEAR`、`HAS_CHUNK`、`MENTIONS` 四类关系。
- TF-IDF召回与图关系加权结合，实现Graph RAG检索。
- 支持单公司、多公司对比、年份和金融指标理解。
- 固定10道 `eval_questions.md` 问题，也支持自由提问。
- 网页展示答案、来源文本、页码、查询理解和SVG知识图谱。
- 配置DeepSeek时生成简洁答案；未配置时返回本地证据摘要。

## 使用本地已有环境

项目不需要新建或修改环境，直接使用：

```text
C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe
```

## 1. 填写DeepSeek API（可选）

打开项目根目录 `.env`：

```env
DEEPSEEK_API_KEY=
```

把Key填写在等号后面即可。留空也能运行，页面会显示Graph RAG检索到的证据摘要。

## 2. 第一次运行：建立索引

在项目目录运行：

```powershell
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" prepare.py
```

程序默认读取：

```text
C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition.zip
```

如果附件移动了，可设置环境变量 `CCF_ZIP_PATH` 指向新的ZIP路径。

本次真实附件建库结果：4705个文本块、4734个节点、5467条关系。

## 3. 启动网页

可以在PyCharm中右击运行 `app.py`，或执行：

```powershell
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" app.py
```

浏览器打开：<http://127.0.0.1:8019/>

## 4. 使用方式

1. 从“固定评测问题”选择一道题，或直接输入问题。
2. 设置Top K，点击“开始 Graph RAG 检索”。
3. 查看回答、查询理解、命中文本块和知识图谱。
4. 点击图谱节点可查看节点类型和元数据。

## 主要文件

- `prepare.py`：读取ZIP并建立Graph RAG索引。
- `corpus.py`：识别年报、解析JSONL、按页切块。
- `graph_rag.py`：构建知识图谱、图增强检索、相关子图裁剪。
- `questions.py`：解析 `eval_questions.md`。
- `app.py`：FastAPI接口和8019端口启动入口。
- `index.html`：问答与知识图谱网页。
- `tests/`：题库、语料、图检索、网页和真实数据测试。

## 测试

```powershell
& "C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe" -B -m unittest discover -s tests -v
```

`data/`、原始附件、缓存和真实API Key不会提交Git。

