# RAG 评测与优化

## 1. RAGAS 指标

建议构建一组覆盖每个角色和风险场景的样本集：问题、标准答案、召回上下文、ground truth、角色类别、难度、数据版本。

- Faithfulness：回答是否被召回上下文支持。
- Context Precision：高相关上下文是否排在前面。
- Context Recall：是否召回了回答所需的信息。
- Answer Relevancy：回答是否真正回应问题。
- Answer Correctness：与人工参考答案的一致程度。

接口：`POST /api/v1/evaluate`。

默认 `RAGAS_ENABLED=false`，接口返回离线可运行的本地代理指标，不会产生外部模型调用。配置 OpenAI 兼容模型和 `LLM_API_KEY` 后，将 `RAGAS_ENABLED=true`，接口会执行真实 RAGAS `Faithfulness` 和 `ContextRelevance` 评测；评测失败时仍会记录完整异常并回退到本地代理指标。

## 2. 优化顺序

1. 数据清洗：去重、去水印、删除空白页、修复编码、保留标题和页码。
2. 分块优化：对标题、段落、列表、表格分别处理；比较 300/600/1000 字符和不同 overlap。
3. Query 改写：补全指代、省略和专业术语；必要时做多查询扩写。
4. 多路召回：向量、BM25、结构化数据库、知识图谱、互联网搜索分别取 Top-K，再做 RRF 或加权融合。
5. 重排：使用 BGE-Reranker；对高风险领域设置更严格的分数阈值。
6. 上下文压缩：去重、摘要、父子块合并，控制 Prompt 长度。
7. 生成优化：模板约束、引用标注、回答校验、正则后处理、拒答策略。
8. 系统优化：缓存、批量 embedding、异步任务、连接池、流式输出、硬件量化、Nginx 负载均衡。

## 3. 压测指标

使用 JMeter 或 Locust 关注：QPS、平均延迟、P50/P95/P99、首 token 延迟、完整回答延迟、错误率、并发连接、CPU/GPU/显存、Redis/Milvus/MySQL 连接池。

建议分开测试：

- 纯 `/api/v1/search` 检索性能。
- Mock LLM 下的端到端 RAG 性能。
- 真实模型下的首 token 和吞吐。
- 流式连接在慢客户端、断开和高并发下的资源回收。

## 4. 测试分层

- 单元测试：分块、tokenize、混合召回、重排、后处理、记忆顺序。
- 集成测试：SQLite + 本地索引 + Mock LLM，验证上传到聊天的完整链路。
- 接口测试：Postman/Apipost 保存环境变量和回归集合。
- 压力测试：JMeter 逐级增加并发，记录错误和性能曲线。
- 人工评审：高风险角色检查事实、引用、边界提示和拒答质量。
