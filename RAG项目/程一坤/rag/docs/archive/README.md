# archive/ 归档说明

> **本目录内容已作废，不作为开发依据。** 仅用于追溯"当时是怎么想的"。

| 项 | 内容 |
|---|---|
| 归档日期 | 2026-09-16 |
| 归档原因 | 历史计划之间路径与结构互相矛盾（同时存在 `app/database/`、`app/importers/`、`app/indexing/`、`app/search/` 四套说法），导致"计划 vs 代码"长期不一致，且出现把计划当成已实现代码的误判 |
| 替代文档 | `docs/开发路线图.md`（唯一权威）、`docs/目录与命名约定.md`（目录与命名唯一权威） |

---

## 归档清单

### plans/（8 份历史计划）

| 文件 | 当时目的 | 现状 |
|---|---|---|
| 2026-09-14-legal-rag-mvp.md | 首期 MVP 总计划 | 已被路线图取代 |
| 2026-09-15-legal-rag-p0-corrections.md | 采集可靠性修复 | 已完成，结果并入路线图阶段 0 |
| 2026-09-15-legal-rag-offline-pipeline-mysql-import.md | 离线数据包 + MySQL 导入 | 大部分已完成 |
| 2026-09-16-labor-law-offline-ingest.md | 劳动法离线入库 | 已完成，遗留问题见路线图阶段 2 |
| 2026-09-16-legal-rag-ordered-roadmap.md | 收敛路线（7 阶段 8 任务） | 已被路线图取代 |
| 2026-09-16-hybrid-search.md | 混合检索计划 | **未执行（0/20 步骤）**，内容已并入路线图阶段 3 |
| 2026-09-16-auth-and-persistence.md | 认证持久化计划 | 未执行，并入路线图阶段 0 与后续 |
| 2026-09-16-memory-system.md | 记忆系统计划 | 未执行，并入路线图阶段 5 |

### specs/（2 份设计规格）

| 文件 | 现状 |
|---|---|
| 2026-09-15-legal-rag-mysql-offline-pipeline-design.md | 已实现，设计要点保留在本文档与路线图 |
| 2026-09-15-legal-rag-p0-corrections-design.md | 已实现 |

---

## 使用规则

```text
1. 不得把 archive/ 里的"创建：xxx.py"当作已存在的代码（已发生过一次误判）
2. 不得按 archive/ 里的路径写新代码——路径一律以 docs/目录与命名约定.md 为准
3. 需要历史设计意图时，可以查阅，但结论必须回到 docs/开发路线图.md
4. 本目录不再修改
```

---

**文档结束** ｜ 2026-09-16
