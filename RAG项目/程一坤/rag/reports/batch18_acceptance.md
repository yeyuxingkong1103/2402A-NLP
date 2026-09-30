# 批次 18 验收报告：fusion.py 拆分 + 注释补齐

日期：2026-09-21

## 改动清单
- `backend/app/retrieval/fusion.py`：只留 RRF 融合（RRF_K / RankedItem / fuse_results），157→162 行
- `backend/app/retrieval/parent_collapse.py`：新建，父块解析与归并（load_keyword_parent_items / collapse_parent_items / _fetch_rows / _as_parent_item / _rank_score / _prefer_score），250 行
- `backend/app/retrieval/service.py`：import 行 15 改为 `from app.retrieval.parent_collapse import collapse_parent_items, load_keyword_parent_items`
- `backend/tests/test_parent_collapse_order.py`：import 与 monkeypatch 目标改为 `app.retrieval.parent_collapse._fetch_rows`
- 临时脚本（项目根，不进 backend/）：`b18_fusion_original.py`（拆分前备份）、`b18_verify.py`（校验脚本）

## 验收

### a) 行数与注释覆盖率（# 注释行 + docstring 行 / 总行）
```
fusion.py:          总行=162 (≤250 OK)  注释+docstring=49 行  覆盖率=30.2% (≥30% OK)
parent_collapse.py: 总行=250 (≤250 OK)  注释+docstring=75 行  覆盖率=30.0% (≥30% OK)
```

### 附加证明：逻辑一行未改（剥 docstring 后 AST 逐函数/类/顶层赋值比对）
```
新版对象总数: 10  缺失: []  不一致: []  新增赋值: 0
PASS
```

### b) find app -name '*.py' -exec wc -l {} + | awk '$1 > 300'
**未达标——但超标文件全部是本批之前的既有文件**，本批仅新增/改动的两个文件均 ≤250：
```
   516 app/chat/service.py
   318 app/db/import_service.py
   324 app/db/sql_models.py
   383 app/ingest/chunker.py
   309 app/ingest/law_metadata_extractor.py
   307 app/ingest/offline_ingest.py
   391 app/memory/long_term.py
   327 app/retrieval/service.py
   301 app/review/review_service.py
```
批次 17-2 的"单文件 ≤300 行"约束只覆盖当时 5 个核心链路文件，其余文件从未按此规则
压过行数。按"有缺口先报告、不擅自扩范围"处理：未动这 9 个文件。需你裁决：
(A) 把 ≤300 规则扩到全 app/，另开批次拆分；(B) 验收 b 收窄为"本批触及的文件 ≤300"。

### c) 全量测试
```
413 passed, 5 warnings in 9.90s
```

### d) demo_ask "经济补偿怎么算"
```
【检索统计】向量召回 11 / 关键词召回 20 / 融合后 27 / 重排后 5
【引用法源】共 5 条（劳动合同法 47/97 条、实施条例 27/25/10 条）
【护栏】citation_check_passed, guardrails_applied
```
回答完整（年限×月工资、上限三倍封顶 12 年、赔偿金互斥等），引用标注正常。

## 备注
- 术语与 docs/CONTEXT.md 一致：article_number / paragraph_number / item_number。
- 零新增依赖；docs/ 未动。
- 教训（已多次踩）：Edit 工具并行多次调用会静默丢盘，本批因此两次返工；后续一律单条
  Edit 后立即 grep 复核。
