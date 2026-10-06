# Specification Quality Checklist: 提问落盘与问题向量化（S8）

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-27
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

### 第 1 轮校验（2026-09-27）— 未通过

唯一未通过项：`No [NEEDS CLARIFICATION] markers remain`，存在 3 处。三处都不是"没想清楚"，而是**与既有设计存在真实冲突或存在真实的成本分叉**：

| 位置 | 标记 | 冲突对象 / 分叉点 |
|---|---|---|
| FR-003 | 留存期限未定 | `docs/01` §7.2 明列「MVP 不做**历史记录**」，理由「少收集个人信息也降低合规负担」。本特性要做的正是持久化用户问题原文 |
| FR-011 | 向量化触发位置未定 | 服务端同步做（模型 2.27 GB 常驻）vs 独立脚本批量做（服务端本期不介入）。`docs/02` §10 的部署视图按"常驻内存"绘制，但用户本次的措辞是"生成脚本" |
| FR-021 | 落盘形态未定 | 一问题一文件 / 共享追加文件 / 内联在记录行内，三者对可追溯性与文件数量的影响不同 |

**三处均涉及与既有文档的关系，按用户"严禁私自做决策"的约定，MUST 由人工裁决。**

### 已通过项的两点说明（记录以免后续误改）

1. `No implementation details` 记为通过，但规格中确实出现了模型路径 `E:\资料\BAAI--bge-m3`、`data/` 目录、`backend/` 目录、1024 维 / float32 / CLS / L2 这些具体参数。理由与 `specs/006` 的同类判断一致：这些**不是实现自由选择** ——
   - 模型路径与编码口径由用户直接指定，且是"向量空间必须一致"这一硬约束的**具体内容**（FR-009/FR-014 要验证的正是它们与索引侧一致）；
   - 目录与 300 行上限来自 constitution 与用户明确要求。

2. **本规格刻意没有规定"用什么库加载模型"** —— `backend/embed/model.py` 已用 `transformers` 实现了这件事，本特性只需复用（FR-009）。规格层面只锁定"口径"，不锁定"实现载体"。

### 第 2 轮校验（2026-09-27）— 全部通过

三处标记已由用户裁决并回填为正式需求：

| 原标记 | 裁决 | 落点为 |
|---|---|---|
| FR-003 留存期限 | 全文留存 + 三条约束 | FR-001、FR-028、FR-029、FR-030 |
| FR-011 向量化触发位置 | 服务端在 `POST /ask` 内同步完成 | FR-031、FR-032 |
| FR-021 落盘形态 | 单文件追加 JSONL，向量内联 | FR-033、FR-034 |

回填后重新逐条核对，全部通过。

**三点记录，供 plan 阶段与后续评审引用**：

1. **D1 使本特性的改动面从"新增一个脚本"扩大到"修改正在运行的服务"** ——
   `backend/serve.py` 启动流程、`backend/api/` 调用点都要动，且服务内存占用会因
   2.27 GB 权重常驻而显著上升。plan 阶段 MUST 为这一影响留出验收项（启动时间、
   内存占用），不能只验收"向量算出来了"。

2. **D3 的三条约束是本特性的交付物，不是可选项** —— 尤其 FR-030（`docs/01` §7.2
   加注记）。若只在代码里留存、不去改那份文档，那么"文档说不做历史记录、代码在
   做留存"这个矛盾会原样保留下来，后来人无从判断哪个算准。

3. **FR-014~FR-018 的指纹门禁 MUST 在本期落地** —— 本期向量没有任何消费者，
   漂移不会立刻暴露。这是唯一一件"事后无法补救"的事。

### 一处需要在 plan 阶段特别注意的事

本特性交付的向量**本期没有任何消费者**（检索模块未实现）。这意味着：

- 向量空间漂移的代价**不会立刻显现**，只会在检索接入时集中爆发；
- 因此 FR-014~FR-018 的指纹门禁 **MUST 在本期就落地并被验收**，不能"等检索接入时再说"——等到那时，磁盘上已经积压了一批口径不明的向量，且无从判断哪些是错的。

**若 plan 阶段认为门禁可以推后，应回到本规格重新裁决，而不是静默省略。**
