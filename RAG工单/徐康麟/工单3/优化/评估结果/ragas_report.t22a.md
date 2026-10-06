# RAGAS 评估报告

## ⚠️ RAGAS 未运行（依赖不可用，本机断网）

> 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化　生成时间：2026-10-04T20:14:54+0800　运行标记：`t22a`
> **本报告不含任何 RAGAS 数值**：`ragas`（及其 `langchain*` 依赖）在本机未安装且**断网无法安装**，
> 四项指标（faithfulness / answer_relevancy / context_precision / context_recall）一律为 `null`。
> 与工单2 基线对齐：`工单2/优化/基线/baseline_metrics.json` 的 `ragas` 字段同样是
> 「未运行（依赖不可用且本机断网）」、`eval_records` 中四项均为 `null` → **前后两次都未运行 RAGAS**，
> 该口径一致，不存在「基线有、优化后没有」的缺失。

## 1. 依赖探测（真实执行，失败即如实记录）

| 包 | import 结果 | 说明 |
| --- | --- | --- |
| `ragas` | ❌ 不可导入（ModuleNotFoundError/其它） | ModuleNotFoundError: No module named 'ragas' |
| `langchain_openai` | ❌ 不可导入（ModuleNotFoundError/其它） | ModuleNotFoundError: No module named 'langchain_openai' |
| `langchain_community` | ❌ 不可导入（ModuleNotFoundError/其它） | ModuleNotFoundError: No module named 'langchain_community' |
| `datasets` | ❌ 不可导入（ModuleNotFoundError/其它） | ModuleNotFoundError: No module named 'datasets' |
| `pandas` | ❌ 不可导入（ModuleNotFoundError/其它） | ModuleNotFoundError: No module named 'pandas' |

## 2. 确定性替代指标（本工单实际使用，全部可复算）

| 替代指标 | 值 | 对应的 RAGAS 关注点 | 计算口径 |
| --- | --- | --- | --- |
| 准确率（14 题，工单1 Evaluator 口径） | 14/14 = 100.0% | answer_relevancy / faithfulness | 五步确定性判分（子串/带单位数值/实体/比例/二元组相似度 ≥0.62） |
| 召回命中率（证据原文落在返回块） | 14/14 = 100.0% | context_recall | `is_evidence_hit(chunks, evidence_verbatim)` |
| 引用可回溯（四点校验） | 14/14 = 100.0% | faithfulness（引用可核） | 页码范围 + 文件在语料 + 块可查且页一致 + 引用处有支撑原文 |
| 「不清楚」正确率（无关问题集） | 6/6 = 100.0% | context_precision（负例侧） | 无关问题必须回「不清楚」且不得含禁用串 |
| 首字延迟 max / p95 | 343.21 ms / 343.21 ms | ——（时延，RAGAS 不覆盖） | 逐题 LLM 首 token（预热后口径） |

## 3. 未运行 RAGAS 的影响与替代理由

1. **为什么不用 RAGAS**：`ragas` 需要 `langchain*` / `datasets` / 联网下载指标提示词与 judge 模型，本机无网络、无对应 wheel，装不上（探测证据见第 1 节）。
2. **为什么替代指标足够**：RAGAS 的四项本质是「答案是否被上下文支持 / 上下文是否召回了答案依据」，本工单用**可复算的确定性判据**覆盖同一关注点：`is_evidence_hit` 判定证据原文是否落在返回块（= recall），`answer_support_check` 判定答案能否在引用处核实（= faithfulness 的可核版本），答案与金标准的五步判分（= answer_relevancy 的确定性代理）。
3. **诚实声明**：以上替代指标与 RAGAS 的定义**不等价**，不能相互换算；任何把替代数值标注为 RAGAS 数值的做法都是伪造，本报告不做。

