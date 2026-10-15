# Specification Quality Checklist: 入库（S6 步骤）

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

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`

### 校验记录

**第 1 轮（2026-09-27）**：未通过项 **No [NEEDS CLARIFICATION] markers remain**（4 项待裁决）。

这 4 项是**有意保留**的，不是规格缺陷。理由与 `specs/003`/`specs/004` 的既有做法一致：
用户明确要求「遇到不明确的情况告诉用户并推荐 2–3 个方案，严禁私自决策」，因此这些标记
就是"系统没有私自决策"的**证据**，必须保留到用户答复为止。

- **Q1** `source_hash` 的取值来源与删除键（对应「发现的 A」）
- **Q2** `pipeline_config_hash` 的参数集合与取值（对应「发现的 B」）
- **Q3** 未指定文档标识 / collection 不存在时的处置
- **Q4** `index_manifest.json` 的写入职责与多文档合并语义

**第 2 轮（2026-09-27）**：**17/17 全部通过。**

四项裁决结果（全部选 A）已固化为 spec 文末的 **D1–D4**，[NEEDS CLARIFICATION] 标记已全部
替换为具体条款（FR-002 / FR-008 / FR-014 / FR-019 / FR-019a / FR-022 / FR-022a / FR-009a）。

「Content Quality / No implementation details」一项的说明：spec 中出现的
`chunk_id`、`float32[65,1024]`、`http://localhost:19530`、`pymilvus` 等并非实现选择，
而是**上游已锁定的事实**（`docs/04 §9.1`、`specs/004` 的产物、用户给定的连接地址）。
本规格未自行选择任何技术栈。

### 由裁决派生出的文档回写（D1/D2 要求，属于本特性的实现范围）

- `docs/04 §10.1`：更正 `source_hash` 与 `doc_id` 的定义（当前说法与实测矛盾）。
- `docs/04 §7`：把已作废的「300–500 字 + 禁止跨章节合并」更正为实际生效的 300–700 与有限跨章节合并。
- `docs/04 §9.3`：`pipeline_config` 示例的 `chunk_max_chars: 500` → **700**。

### 待回写文档（由 Q2 裁决一并决定）

- `docs/04 §7`：仍写着已被 S4 的 D1 作废的「300–500 字」与「禁止跨章节合并」。
- `docs/04 §9.3`：`pipeline_config` 示例中的 `chunk_min_chars: 300` / `chunk_max_chars: 500` 同为旧值。
- `docs/05 §5`：描述的 `backend.pipeline <子命令>` 统一调度器不存在，实际是各步独立入口。
