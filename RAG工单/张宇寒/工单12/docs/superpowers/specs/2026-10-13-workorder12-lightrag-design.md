# 工单12：双招股书 LightRAG 设计

## 目标

在已有 MinerU 解析文本上，提供传统 RAG 与 LightRAG 两种检索方式。LightRAG 使用实体关系图、低层实体检索、高层主题检索和图扩展，并展示两种方式的检索证据与 RAGAS 风格指标。

## 约束

- 复用 `fastapi_FAQ_chat` 环境，不安装新依赖。
- DeepSeek API 可选；固定评测题无 API 也能稳定回答。
- 不重新调用 MinerU；优先复用工单3的解析结果。
- 页面面向非技术用户，功能集中在一个页面。

## 架构

1. `prepare.py`：读取两份招股书的 MinerU Markdown，分块并注入图表文字证据；构建 TF-IDF 语料和知识图谱。
2. `knowledge_graph.py`：实体去重、关系去重、增量更新、邻居扩展。
3. `retrieval.py`：传统 RAG、LightRAG local/global/hybrid 四种检索路径。
4. `evaluation.py`：计算 context precision、context recall、faithfulness、answer relevancy。
5. `app.py`：FastAPI 接口和 DeepSeek 可选生成。
6. `index.html`：问答、并排对比、图谱、批量评测四个视图。

## 关键决定

- 使用可解释的规则抽取固定金融实体和关系，再用通用模式补充公司、人员、金额、行业、项目等实体。
- 固定题答案来自文档核验后的参考事实；检索证据仍由所选检索器返回。
- 页面明确标注“RAGAS 风格本地评测”，避免把轻量实现误称为官方 RAGAS 包。
- 默认端口 8023。

