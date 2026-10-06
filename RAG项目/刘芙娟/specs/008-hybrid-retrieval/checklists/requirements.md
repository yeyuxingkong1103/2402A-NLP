# Specification Quality Checklist: 混合检索（语义 + BM25 关键词）（S9）

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-28
**Feature**: [spec.md](../spec.md)

## Content Quality

- [ ] No implementation details (languages, frameworks, APIs)
- [X] Focused on user value and business needs
- [X] Written for non-technical stakeholders
- [X] All mandatory sections completed

## Requirement Completeness

- [X] No [NEEDS CLARIFICATION] markers remain
- [X] Requirements are testable and unambiguous
- [X] Success criteria are measurable
- [X] Success criteria are technology-agnostic (no implementation details)
- [X] All acceptance scenarios are defined
- [X] Edge cases are identified
- [X] Scope is clearly bounded
- [X] Dependencies and assumptions identified

## Feature Readiness

- [X] All functional requirements have clear acceptance criteria
- [X] User scenarios cover primary flows
- [X] Feature meets measurable outcomes defined in Success Criteria
- [ ] No implementation details leak into specification

## Notes

- 一项标记为未通过，**刻意为之**：

  **No implementation details** — 本项目既有 specs（001–007）的一致约定是：规格中**必须**点名上游模块路径与契约文档章节，因为"不要重新实现一遍"是本项目最反复出现的失效模式（S8 整篇规格都在防它）。若按通用模板把路径抹掉，规格将无法约束实现者去复用 `backend/query/service.py` 已有的查询向量。此偏离在 plan 阶段的 Constitution Check 中可复核。

- **裁决已完成（2026-09-28）**：Q1 → Milvus 回捞 + jieba；Q2 → RRF（k=60，对外 `score` 保余弦）；Q3 → 新增「检索就绪、生成未就绪」过渡文案。三项已作为 D1/D2/D3 写入 spec.md 并回填 FR-006 / FR-007 / FR-009 / FR-010 / FR-016。三处 `[NEEDS CLARIFICATION]` 标记已全部消除。

- 其余条目在 spec 写作过程中已自查通过；`docs/05` §4.2 契约的四条关键约束（阈值判定在内、空集不抛异常、参数显式传入、`below_threshold` 独立）已逐条落为 FR-011~FR-013。
