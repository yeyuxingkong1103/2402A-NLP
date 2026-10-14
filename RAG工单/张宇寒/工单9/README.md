# 工单9：GraphRAG 优化与评测

本项目基于工单8的金融年报 GraphRAG，增加参考答案对照、四层优化、优化前后检索对比和可重复评测。

## 已实现

- 使用 `sample_questions.pdf` 的4类参考题进行评测。
- 同时运行工单8式 TF-IDF 基线检索与工单9优化 GraphRAG 检索。
- 从 Prompt、实体关系、检索、生成四个层面进行优化。
- 展示 Context Precision、Context Recall、指标变化和响应时间。
- 展示回答、参考答案、查询改写、知识图谱、基线证据和优化证据。
- 支持自定义问题；没有参考答案时不会生成虚假评测分数。
- DeepSeek API 可选；未配置或调用异常时自动返回本地证据摘要。

## 本地环境

继续使用已有环境，不要重新安装：

```text
C:\Users\lenovo\anaconda3\envs\fastapi_FAQ_chat\python.exe
```

如需 DeepSeek 生成答案，在 `.env` 中填写 `DEEPSEEK_API_KEY=`。留空也能完整运行检索、知识图谱和评测。

## 启动

索引已经随项目提供，直接右击运行 `app.py`，然后访问：

```text
http://127.0.0.1:8020/
```

如果需要重新从 `ccf_competition.zip` 建索引，先运行 `prepare.py`，再运行 `app.py`。

## 评测口径

- Context Precision：采用排序感知的平均精度，越相关的文本块排在前面得分越高。
- Context Recall：参考答案事实中，被检索证据覆盖的比例。
- 达标线：Precision >= 0.8、Recall >= 0.9、单题总耗时 < 3秒。
- 当前真实评测：4题平均 Precision 1.00、Recall 1.00，全部单题低于3秒。

详细结果见 `evaluation_report.json`。运行“评测全部参考题”可在网页重新计算。

## 主要文件

- `app.py`：FastAPI 服务、DeepSeek生成和评测接口。
- `graph_rag.py`：基线检索、优化检索和知识图谱子图。
- `evaluation.py`：参考题解析、查询改写和指标计算。
- `sample_questions.md`：从参考PDF整理的可重复评测数据。
- `index.html`：优化前后对比网页。
- `evaluation_report.json`：真实数据评测报告。
- `tests/`：自动化测试。
