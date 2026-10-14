# 工单12：双招股书 LightRAG

这是一个不修改现有环境的轻量实现，复用工单3的 MinerU 解析结果，对两份招股说明书提供传统 RAG 与 LightRAG 对比。

## 启动

1. 使用已有 Conda 环境：`fastapi_FAQ_chat`。
2. 首次重建数据时运行 `prepare.py`；已附带数据则可跳过。
3. 右击运行 `app.py`，或双击 `start.bat`。
4. 浏览器打开 `http://127.0.0.1:8023/`。

## 页面功能

- 智能问答：传统 RAG / LightRAG，支持 local、global、hybrid。
- 对比实验：同一问题并排展示证据、图路径和指标。
- 知识图谱：实体类型、关系类型及来源可视化。
- 批量评测：16 道固定题的 context precision、context recall、faithfulness、answer relevancy。

## 说明

- `.env` 中可填写 DeepSeek API；不填写也可运行固定题和评测。
- 指标为透明可复现的“RAGAS 风格本地评测”，不依赖外部 RAGAS 服务。
- 图表型答案以人工核验后的文字证据补入语料，避免 PDF 图片解析遗漏。
