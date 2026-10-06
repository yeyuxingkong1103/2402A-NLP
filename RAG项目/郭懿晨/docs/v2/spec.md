# v2 检索重排 — 功能规格说明

**Status**: Draft
**Created**: 2026-09-01
**Input**: 在 v1 RAG 检索链路上新增「粗排 + 精排」两阶段重排，可配置、可迭代，按 recall/MRR/延迟对比 v1 基线。
**相关设计文档**: `docs/superpowers/specs/2026-09-01-v2-retrieval-rerank-design.md`

---

## User Scenarios & Testing

### User Story 1 — 问答引用质量提升（Priority: P1）

作为使用 RAG 问答系统的用户，我希望系统在召回后先经过粗排、再经过精排，以便最终给出的引用是更相关的文档片段。

**Why this priority**: v1 中 RRF 检索结果直接取 `top_k=6` 进引用，没有额外重排，低质片段可能占用引用名额、稀释回答依据。这是 v2 的核心价值。

**Independent Test**: 同一评测集分别以 v1 与 v2 方式检索，v2 的 recall/MRR 不退化且引用排序更优，可脱离端到端问答单独验证。

**Acceptance Scenarios**:

1. **Given** 用户提交一个问题，**When** 系统执行检索，**Then** 检索链路为「RRF 召回(候选扩大) → 粗排 → 精排」，最终进入引用的结果为 `final_top_k` 条
2. **Given** `rerank_enabled=false`，**When** 用户提交问题，**Then** 检索行为与 v1 完全一致（搜索 limit=`final_top_k`、按 RRF 分数做 `min_retrieval_score` 过滤）
3. **Given** 粗排后仍无结果或过滤后为空，**When** 系统继续，**Then** 走现有兜底回答「知识库中未找到相关依据」

### User Story 2 — 检索链路可配置、可迭代（Priority: P1）

作为维护者，我希望候选规模、各层 top_k、阈值、粗排打分方式、精排模型路径都是配置项，以便后续按评测指标迭代优化，而不用改代码。

**Why this priority**: 宪法要求粗排/精排按 recall/MRR/延迟可迭代优化；参数必须可调，否则每次实验都要改代码。

**Independent Test**: 修改配置项后无需改业务代码即可改变链路行为，可通过单测验证配置到行为的映射。

**Acceptance Scenarios**:

1. **Given** 修改 `AppSettings` 中任一规模/阈值/打分方式配置，**When** 链路执行，**Then** 按新配置运行
2. **Given** 粗排打分方式可选 `dense_cosine` / `rrf_score` / `hybrid`，**When** 切换配置，**Then** 粗排行为随之变化，无需改代码
3. **Given** 检索返回结果缺失稠密向量（内存兜底路径），**When** 执行粗排，**Then** 粗排回退 `rrf_score` 打分，不报错

### User Story 3 — 可评测、可对比（Priority: P2）

作为维护者，我希望同一评测集能分别产出 v1 与 v2 的检索指标（recall/MRR/延迟），以便判断本轮优化是否有效、为下一轮迭代提供依据。

**Why this priority**: 宪法 §4 要求完整规格链迭代必须产出指标对比；无对比就无法证明 v2 的价值。

**Independent Test**: 运行评测脚本并指定 variant，产出对应版本指标与对比报告，可离线验证。

**Acceptance Scenarios**:

1. **Given** 评测集存在，**When** 以 `--variant v2` 运行评测，**Then** 产出各套 recall/MRR 与平均检索延迟
2. **Given** v1 基线已存档于 `eval/baseline/`，**When** 对比，**Then** 报告 v2 相对 v1 基线的 delta（recall/MRR/延迟）
3. **Given** 每轮 v2 评测完成，**When** 记录结果，**Then** 原始结果落盘 `eval/results/v2_metrics.json`，并追加到唯一指标对比文档 `docs/指标对比/优化指标对比.md`

### Edge Cases

- 用户问题为空 → 走现有兜底回答
- RRF 召回候选为空 → 走现有兜底回答
- 粗排后候选为空 → 走现有兜底回答
- 精排分数过滤后为空 → 走现有兜底回答
- 精排模型路径不存在 / 加载失败 / 打分异常 → 返回可诊断错误，**不允许静默降级**成错误排序的引用
- 检索结果缺失稠密向量 → 粗排回退 `rrf_score` 打分
- `bge-reranker-v2-m3` 与当前 `FlagEmbedding==1.3.3` 不兼容 → 升级 `requirements.txt` 锁定版本并记录原因
- 精排 `final_top_k` 与粗排 `coarse_top_k` 大小关系异常（如 final ≥ coarse）→ 链路按 `min(final, coarse)` 收敛，不崩溃

---

## Requirements

### Functional Requirements

- **FR-001**: 系统在 RRF 召回后执行粗排，从 `retrieval_candidate_k` 候选中保留 `coarse_top_k` 条
- **FR-002**: 系统在粗排后执行精排，用本地 `bge-reranker-v2-m3` 交叉编码器从 `coarse_top_k` 中保留 `final_top_k` 条
- **FR-003**: 粗排复用 BGE-M3 稠密信号（dense 余弦相似度），打分方式可配置为 `dense_cosine` / `rrf_score` / `hybrid`
- **FR-004**: 精排分数为 0~1（sigmoid normalize），作为最终 `min_retrieval_score` 兜底过滤与引用分数
- **FR-005**: `rerank_enabled=false` 时完全退化为 v1 行为，用于 A/B 对比
- **FR-006**: 候选规模、各层 top_k、阈值、打分方式、模型路径、批大小、设备全部通过 `AppSettings` 配置
- **FR-007**: 评测脚本支持 `--variant v1|v2`，产出各套 recall/MRR 与检索延迟，并生成 v1/v2 对比报告
- **FR-008**: 精排模型必须为本地离线加载的 bge-reranker（`bge-reranker-v2-m3`），禁止走 Ollama
- **FR-009**: 精排模型加载失败或打分异常时返回可诊断错误，不允许静默降级

### Non-Functional Requirements

- **NFR-001**: 检索延迟（召回 + 粗排 + 精排，不含 LLM 生成）逐条记录，纳入评测对比
- **NFR-002**: 新增/变更依赖版本在 `requirements.txt` 精确锁定（禁止浮动版本）
- **NFR-003**: 遵循宪法编码规范：类型注解完整、容器类型精确、`snake_case`/`PascalCase`、中文 docstring、魔法数字提取命名常量、禁止硬编码路径
- **NFR-004**: 模型与向量库访问限定在获准路径（`D:\八维学习\bge-reranker-v2-m3` 只读），不越界读写

### Key Entities

- **SearchResult**: 检索结果，含 chunk、score、可选 dense 向量（用于粗排）
- **RetrievalChain**: 编排召回 → 粗排 → 精排 → 阈值过滤的整条链路
- **EvalRunSummary / 对比报告**: 各版本 recall/MRR/延迟指标与 v1 基线 delta

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: 同一评测集 v2 的 recall/MRR ≥ v1 基线（不退化），并在合理参数组合下实现提升
- **SC-002**: 每轮 v2 评测产出真实机器结果落盘 `eval/results/v2_metrics.json`，并追加到 `docs/指标对比/优化指标对比.md`
- **SC-003**: 真实 `bge-reranker-v2-m3` 模型加载与打分测试通过（不 skip）
- **SC-004**: `rerank_enabled=false` 时检索结果与 v1 完全一致
- **SC-005**: 全部相关单元测试/集成测试通过

---

## Assumptions

- GPU 可用，精排走 CUDA + fp16（用户已确认）
- `bge-reranker-v2-m3` 已下载至 `D:\八维学习\bge-reranker-v2-m3`（用户已确认）
- v2 直接修改 v1 代码，不保留并行分支（用户已确认）
- 评测集沿用现有 3 套（检索15 / 生成10 / 安全5）；**safety 集 5 条低于宪法「≥10」下限，本轮沿用以保证与 v1 基线可比，下一轮扩充并在指标对比文档注明**
- 本轮不修复 MinerU 空结果问题（另开一轮）
- 依赖 FlagEmbedding 若与 v2-m3 不兼容，允许在 plan 阶段升级锁定版本
- **前端「检索得分」标签不变**：v2 下 `citation.score` 数值语义变为精排相关性分（0~1），标签仍准确，无需改前端代码
