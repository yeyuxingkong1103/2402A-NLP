# Specification Quality Checklist: MinerU PDF 解析（S2 解析步骤）

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-22
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

### 澄清记录（本轮已由用户裁定，故无遗留 [NEEDS CLARIFICATION]）

| # | 议题 | 用户裁定 | 落点 |
|---|---|---|---|
| 1 | 解析后端：本机只有 VLM 权重，而 `docs/04 §5/§12` 锁 pipeline 后端 | **C — 两者都支持**，`--backend` 参数切换，默认 VLM（唯一具备权重的后端） | FR-006、FR-007、US3 |
| 2 | 脚本形态：`docs/05 §5` 定义的是 `-m backend.pipeline` 包形式 | **单文件脚本** `backend/parse_pdf.py` | Assumptions、FR-012 |
| 3 | 输入来源：`docs/04 §4` 要求 manifest 准入闸门，但 manifest 不存在 | **不引入 manifest**，输入为 PDF 路径或目录，存储位置写在脚本内 | FR-001、FR-002、Assumptions（已记为有意偏离） |
| 4 | MinerU 版本锁定 | 依赖清单写占位版本号，人工安装实测后回填 | FR-014、Assumptions |
| 5 | 依赖清单位置 | 仓库根 `requirements.txt`（constitution 原则 I 规定的唯一权威来源） | FR-012 |

### 有意偏离（须人工记入 `docs/04`）

- **偏离 `docs/04 §4`（S1 语料准入）**：本特性跳过 manifest 准入闸门。原设计的三道硬性校验——sha256 与文件内容比对、`authority_level` 限 1–2 级、`confirmed_by` 非空——均不实现。**影响面**：低权威语料可能在无人确认的情况下进入知识库，而 constitution 原则 V 与 `docs/01 §8.2` 把人工确认作为入口拦截机制。**处置建议**：在 `docs/04 §4` 补记该偏离，或在 S3 清洗前补一个独立的 `intake` 步骤。

### 待回填的文档 TODO

- `docs/04 §5` 的类型对照表当前按 **pipeline 后端 3.x** 的契约书写。本特性默认走 **VLM 后端**，其类型取值集合需实测后回填（对应 `docs/04 §15` 待验证项 P1）。
- `docs/04 §12` 的 `mineru_backend` 锁定值为 `pipeline`，与本次"默认 VLM"不一致，须同步修订。
