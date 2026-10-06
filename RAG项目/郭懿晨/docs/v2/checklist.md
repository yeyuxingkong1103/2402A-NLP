# v2 检索重排质量检查清单

> 基于 `docs/v2/spec.md` 生成。MUST = 必须通过；SHOULD = 应该通过。用于实现后的代码审查与验收确认。

## 功能验收

- [ ] MUST FR-001：RRF 召回后是否先执行粗排，候选从 `retrieval_candidate_k` 保留到 `coarse_top_k`？
- [ ] MUST FR-002：粗排后是否执行精排，用 bge-reranker-v2-m3 从 `coarse_top_k` 保留到 `final_top_k`？
- [ ] MUST FR-003：粗排打分方式是否支持 `dense_cosine` / `rrf_score` / `hybrid` 三种，且默认 `dense_cosine`？
- [ ] MUST FR-003：候选 dense 缺失时，粗排是否回退 `rrf_score` 打分而不报错？
- [ ] MUST FR-004：精排分数是否为 0~1（normalize），并用作最终 `min_retrieval_score` 过滤与引用分数？
- [ ] MUST FR-005：`rerank_enabled=false` 时检索是否完全退化为 v1（search limit=final_top_k + RRF 分过滤）？
- [ ] MUST FR-006：候选规模/各层 top_k/阈值/打分方式/模型路径/批大小/设备是否全部由 `AppSettings` 配置？
- [ ] MUST FR-007：评测脚本是否支持 `--variant v1|v2|both`，并产出各套 recall/MRR 与延迟？
- [ ] MUST FR-007：是否产出 `v1_vs_v2_comparison.json` 对比报告（recall/MRR 及延迟 delta）？
- [ ] MUST FR-008：精排是否本地离线加载 bge-reranker-v2-m3，未调用 Ollama？
- [ ] MUST FR-009：精排模型路径不存在 / 加载失败 / 打分异常时，是否返回可诊断错误而非静默降级？
- [ ] SHOULD US3-验收：每个 item 是否记录 `latency_s`，summary 是否含 `avg_latency_s`？

## 异常与边界

- [ ] MUST 用户问题为空 → 返回兜底「知识库中未找到相关依据。」
- [ ] MUST RRF 候选为空 → 返回兜底
- [ ] MUST 粗排后候选为空 → 返回兜底
- [ ] MUST 精排过滤后为空 → 返回兜底
- [ ] MUST 空问题走兜底后，`fallback=true` 且 citations 为空
- [ ] SHOULD `final_top_k >= coarse_top_k` 时不崩溃，按 `min(coarse, final)` 收敛

## 配置与契约

- [ ] MUST `top_k` 是否已从配置与 `.env.example` 移除，替换为 `retrieval_candidate_k` + `final_top_k`？
- [ ] MUST 依赖版本在 `requirements.txt` 精确锁定（若升级 FlagEmbedding，需锁定新版本并记录原因）
- [ ] SHOULD 前端「检索得分」标签不变，v2 下 `citation.score` 语义为精排相关性分（0~1）

## 质量与安全

- [ ] MUST 所有公开函数含类型注解与中文 docstring，命名符合宪法（snake_case/PascalCase）
- [ ] MUST 无硬编码路径（模型路径走配置）
- [ ] MUST 模型读取限定在获准路径（`D:\八维学习\bge-reranker-v2-m3`），不越界读写
- [ ] SHOULD 指标对比文档 `docs/指标对比/优化指标对比.md` 追加 v2 记录，且数据来自真实运行

## 验收标准（SC）

- [ ] MUST SC-001：同一评测集 v2 的 recall/MRR ≥ v1 基线（不退化）
- [ ] MUST SC-002：`eval/results/v2_metrics.json` 由真实评测产出
- [ ] MUST SC-003：真实 v2-m3 模型加载与打分测试通过（`test_rerank_real.py`）
- [ ] MUST SC-004：`rerank_enabled=false` 时检索结果与 v1 一致
- [ ] MUST SC-005：全部相关单元/集成测试通过（`pytest backend/tests -v`）

## 宪法合规

- [ ] MUST 需求文档 `docs/需求说明.md` 含 v2 版本头与版本迭代表，FR15~FR18 已加入，「reranker 精排」已从范围外移除
- [ ] MUST `docs/版本迭代.md` 存在并记录 v1/v2
- [ ] MUST `docs/架构/架构图-v2.md` 六层架构图含粗排/精排链路
- [ ] MUST 完整规格链产物齐备：`docs/v2/spec.md`、`docs/v2/plan.md`、`docs/v2/tasks.md`、`docs/v2/checklist.md`
