# results 目录说明

> 工单编号：**人工智能NLP-RAG-Query理解优化任务**
> 本目录存放运行产物（JSON 供程序消费 / 交叉验证，Markdown 供人读与答辩展示）。
> 所有文件均由脚本**运行后生成**，不含手工填写的结果；请勿硬编码或修改数值。

---

## 一、文件清单

| 文件 | 生成脚本 | 关键字段 / 内容 |
|------|----------|------------------|
| `index_stats.json` | `src/build_index.py` | `docs`（两份文档的页数/块数）、`n_chunks`、`n_vectors`、`n_bm25_docs`、`build_seconds`、`chunk_types`、`config` |
| `understanding_demo.json` | `src/query_understanding.py` | `meta`（LLM 是否在线）、`summary`（通过/失败/跳过）、`cases`（逐用例：输入/期望/实际/结果/说明）、`5轮脚本一览` |
| `understanding_demo.md` | 同上 | 人读报告：总体结果表、逐用例表、核心断言详解、五轮理解一览、已知行为与优化方向 |
| `multi_turn_conversation.json` | `src/multi_turn_chat.py` | `meta`（pipeline 配置、总耗时、耗时汇总、LLM 用量）、`turns`（逐轮：问题/改写后/意图/实体/子问题/答案/Top-3 片段/timings/耗时） |
| `multi_turn_conversation.md` | 同上 | 逐轮记录 + 能力对照表 + 结论（含「阶段耗时明细」） |
| `multiturn_ablation.json` | `src/ablation_multiturn.py` | 两组汇总（检索命中率、答案准确率、耗时）、`对比`（提升幅度、失败轮次）、`逐轮对比`（两组的检索式与判定）、`明细` |
| `multiturn_ablation.md` | 同上 | 总体对比表、逐轮对比表、**核心证据**（第 2/3/4 轮为何失败）、结论 |
| `evaluation.json` | `src/run_evaluation.py` | `keyword_accuracy`（准确率与逐轮命中）、`ragas_summary`（RAGAS 指标 + Hit Rate/MRR + 不可用指标）、`latency`、`多语言自检`、`turns` |
| `evaluation_records.json` | 同上（`evaluate.save_report`） | 逐条 `EvalRecord`：question/answer/ground_truth/contexts/retrieved_docs/retrieved_pages/latency/metrics |
| `evaluation.md` | 同上 | 评估报告：核心结论、逐轮明细、RAGAS 指标块、多语言自检、口径说明 |
| `ground_truth.json`（**可选，人工创建**） | 人工 | 覆盖默认判定口径：`{"1": {"answer_keywords": [...], "min_years": 3, "reference_answer": "…全文…"}, ...}` |

> `ground_truth.json` 是唯一需要人工准备的文件（可选）。不创建时使用
> `src/wo05_common.py::TURN_SPECS` 的「最小必要证据」口径，保证离线可复现。

---

## 二、怎么用这些结果

1. **验收「5 轮都答了」**：看 `multi_turn_conversation.md` 的逐轮记录（含检索片段与页码）。
2. **验收「准确率 ≥ 90%」**：看 `evaluation.md` 的「关键词命中式答案准确率」；
   若要更严格口径，先补 `ground_truth.json` 再重跑 `run_evaluation.py`。
3. **验收「3 秒响应」**：看 `multi_turn_conversation.md` 的耗时汇总（3 秒达标率）与每轮阶段明细；
   服务侧延迟看 `/api/metrics`。
4. **证明「提升来自 Query 理解」**：看 `multiturn_ablation.md` 的对比表与「核心证据」章节。
5. **回答「改写逻辑是否可靠」**：看 `understanding_demo.md` 的断言结果（含核心断言详解）。

---

## 三、复现命令

```bash
cd 工单05-Query理解优化-多轮对话/src
python build_index.py                # → index_stats.json
python query_understanding.py        # → understanding_demo.json/.md
python multi_turn_chat.py --fast     # → multi_turn_conversation.json/.md
python ablation_multiturn.py --fast  # → multiturn_ablation.json/.md
python run_evaluation.py             # → evaluation.json / evaluation_records.json / evaluation.md
```

---

## 四、阅读结果时的注意事项

- **耗时受环境影响**：网络延迟、模型负载、是否命中 LLM 磁盘缓存都会影响耗时；
  同一台机器连续跑两次，第二次通常更快（缓存命中），这是预期行为。
- **在线/离线差异**：未配置 `DEEPSEEK_API_KEY` 时，`understanding_demo` 的在线用例显示「跳过」，
  指代消解退化为原问题，消融实验两组的差距会缩小——此时的结果不代表系统真实能力。
- **指标降级**：`ragas_summary` 中若出现「不可用指标」，说明该指标依赖的 LLM 或嵌入模型不可用，
  其余指标仍然有效，报告中会明确标出。
- **判定口径**：默认口径是「最小必要证据」（关键信息点是否出现），
  用于离线自动判定；它比人工逐字比对宽松，但方向一致，适合做回归与对比实验。
