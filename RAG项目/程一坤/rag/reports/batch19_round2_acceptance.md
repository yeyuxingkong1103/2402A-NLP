# 批次 19 第 2 轮验收报告（3 个中度超标文件）

日期：2026-09-21
范围：④ `app/db/sql_models.py`(324) ⑤ `app/retrieval/service.py`(327) ⑥ `app/db/import_service.py`(318)
原则：逻辑零改动搬运；备份先行（`.bak_b19r2/`）；新文件 ≤250 行；不留兼容垫片（④ 聚合入口除外）

---

## 一、新增文件登记（新增文件 → 职责一句话）

| 新增文件 | 行数 | 职责一句话 |
|---|---|---|
| `backend/app/db/document_models.py` | 164 | 文档域实体：Document / DocumentVersion / DocumentChunk / CrawlRecord / ImportRecord |
| `backend/app/db/law_models.py` | 175 | 法规域实体：Law / LawVersion / Article（条/款/项三级结构） |
| `backend/app/db/user_models.py` | 48 | 用户域实体：User（认证 + 长期记忆开关） |
| `backend/app/retrieval/item_convert.py` | 109 | RankedItem 转换纯函数：向量结果→RankedItem、重排结果重建、重排锚点文本 |
| `backend/app/db/import_version_status.py` | 83 | 数据包增量判定：new / unchanged / updated 决策（双比较 content_hash + chunk_fingerprint） |

改动文件：
- `app/db/base.py`（共用件）：新增 `utc_now`（原来在 sql_models 里），成为 Base + utc_now 的单一来源
- `app/db/sql_models.py`：改为**统一导入入口**——只 re-export + `__all__`，零逻辑、零表定义（38 行）
- `app/db/chat_models.py`：docstring 补"实体定义按域分置"注明（注释级改动，零逻辑）
- `app/retrieval/service.py`：266 行（原 327），主链路只剩编排
- `app/db/import_service.py`：277 行（原 318）

### ④ 关于"实体定义分置"的申报（需你登记文档）
- 表定义现在分布在 **5 处**：`base.py`(Base) + `document_models.py` + `law_models.py` + `user_models.py` + **`chat_models.py`（本来就独立的既有文件，本轮未拆）**
- 已在每个域文件 docstring 写明："实体定义按域分置；对外统一入口是 app/db/sql_models.py"，并在 chat_models.py 同步注明"改动表结构时务必确认涉及的所有域文件"
- **30 个调用方 import 一行未改**：聚合入口除 10 张表类外，额外 re-export 了 `Base` 与 `utc_now`（有调用方从 sql_models 取这两个名字，见 `chat_models.py` / `chat_store.py` / `import_service.py` / `test_import_incremental.py`）

---

## 二、验收输出

### a) 行数与注释覆盖率（新文件全部 ≥30%，全部 ≤250 行）
```
base.py                        14 行  docstring  2 + # 注释  0 = 覆盖率 14.3%   （既有共用件，非本轮新文件）
document_models.py            164 行  docstring 24 + # 注释 33 = 覆盖率 34.8%
law_models.py                 175 行  docstring 32 + # 注释 35 = 覆盖率 38.3%
user_models.py                 48 行  docstring 15 + # 注释  9 = 覆盖率 50.0%
sql_models.py                  38 行  docstring 13 + # 注释  0 = 覆盖率 34.2%
service.py                    266 行  docstring 25 + # 注释 18 = 覆盖率 16.2%   （瘦身后的原文件，≤300 即达标）
item_convert.py               109 行  docstring 28 + # 注释  6 = 覆盖率 31.2%
import_service.py             277 行  docstring 16 + # 注释 17 = 覆盖率 11.9%   （瘦身后的原文件，≤300 即达标）
import_version_status.py       83 行  docstring 16 + # 注释  9 = 覆盖率 30.1%
```

### b) AST 零改动比对（剥 docstring / 模块级 import / staticmethod 装饰器，逐对象 dump 比对）
```
[sql_models] PASS  对象数 原 11 → 新 11
    缺失: 无
    不一致(未白名单): 无  白名单: 无
    新增(未白名单): 无  白名单: 无
[retrieval_service] PASS  对象数 原 8 → 新 10
    缺失: 无
    不一致(未白名单): 无  白名单(内联块抽取的编排方法): ['retrieve', '_rerank_fused_candidates']
    新增(未白名单): 无  白名单(抽出模块的新函数/类): ['vector_articles_to_ranked_items', 'rebuild_rerank_items']
[import_service] PASS  对象数 原 10 → 新 12
    缺失: 无
    不一致(未白名单): 无  白名单(内联块抽取的编排方法): ['_import_package_rows']
    新增(未白名单): 无  白名单(抽出模块的新函数/类): ['decide_version_status', 'VersionDecision']

[retrieval_service 抽出块一致性]
  向量转换内联块 == vector_articles_to_ranked_items 函数体: 一致
  重排重建内联块 == rebuild_rerank_items 函数体: 一致
OVERALL: PASS
```
> 说明：⑤⑥ 的白名单差异来自"把内联块抽成独立函数/模块"这一动作本身（你本轮指定的优先方案），
> 编排方法体只发生"内联块 → 一行调用"的替换；被抽出块与新函数体做了独立 dump 比对，逐字一致。
> ④ 无任何白名单差异——纯整对象搬家，11 个对象（Base / utc_now / 10 张表）全部一致。

### c) `grep -rn "class .*Base" app/`（表定义只出现在域文件）
```
app/db/base.py:6:class Base(DeclarativeBase):
app/db/chat_models.py:24:class ChatSession(Base):
app/db/chat_models.py:71:class ChatMessage(Base):
app/db/document_models.py:16:class Document(Base):
app/db/document_models.py:31:class DocumentVersion(Base):
app/db/document_models.py:66:class DocumentChunk(Base):
app/db/document_models.py:92:class CrawlRecord(Base):
app/db/document_models.py:113:class ImportRecord(Base):
app/db/law_models.py:16:class Law(Base):
app/db/law_models.py:62:class LawVersion(Base):
app/db/law_models.py:114:class Article(Base):
app/db/user_models.py:16:class User(Base):
```
（其余匹配均为 Pydantic 的 `*Request(BaseModel)`，与 ORM 无关；`sql_models.py` 里 0 条）

### d) 表结构不变（当前 MySQL 库 DESCRIBE + ORM 互检）
```
== DESCRIBE documents（7 列）==
  id int NO PRI / document_key varchar(64) NO UNI / source_url varchar(768) NO UNI
  title varchar(512) NO / current_version_id int YES / created_at datetime NO / updated_at datetime NO
== DESCRIBE document_versions（14 列）==（含 chunk_fingerprint / version_status / reviewed_by / reviewed_at / review_note）
== DESCRIBE document_chunks（12 列）==（含 parent_chunk_key / article_number / paragraph_number / item_number / retrieval_text）
== DESCRIBE users（9 列）==（含 is_admin / is_active / long_term_memory_enabled）
== DESCRIBE laws（12 列）== / law_versions（12 列）/ articles（11 列）

== ORM(拆分后模型) vs 数据库 列集合互检 ==
  documents 一致（7/7）        document_versions 一致（14/14）   document_chunks 一致（12/12）
  users 一致（9/9）            laws 一致（12/12）                law_versions 一致（12/12）
  articles 一致（11/11）
RESULT: ALL MATCH
```

### e) 全量测试
```
413 passed, 5 warnings in 10.42s
```
（不减：批次 19 第 1 轮同为 413）

### f) demo_ask "经济补偿怎么算"
```
【检索统计】向量召回: 11 条 / 关键词召回: 20 条 / 融合后: 27 条 / 重排后: 5 条
【引用法源】共 5 条（劳动合同法 47/97 条、实施条例 27/25/10 条）
【回答】完整输出，年限与月工资口径正常带引用
```

### g) `>300` 行复查（规则扩到全 app/ 后的剩余）
```
   309 app/ingest/law_metadata_extractor.py
   308 app/ingest/offline_ingest.py
   301 app/review/review_service.py
```
仅剩第 3 轮 3 个文件。

---

## 三、本批结束删除的备份与临时脚本清单
| 项目 | 说明 |
|---|---|
| `.bak_b19r2/`（sql_models.py / retrieval_service.py / import_service.py / base.py） | 第 2 轮拆分前备份，验收全过后删除 |
| `b19_orig_chat_service.py` / `b19_orig_long_term.py` / `b19_orig_chunker.py` | 第 1 轮 AST 基线，使命完成删除 |
| `b19_verify.py` | 第 1 轮校验脚本（第 2 轮已由 b19r2_verify.py 取代） |
| `b19r2_describe.py` | 本轮 describe 校验脚本 |
| 保留 | `b19r2_verify.py`（第 3 轮改造复用，批次 19 结束时删） |

---

## 四、过程记录（踩坑自曝）
1. **并行 Edit 丢盘复发**：`document_models.py` / `item_convert.py` 一次并行 6 个 Edit 只落地约一半。
   后续改为"整文件 Write 定稿"或"单条 Edit + 立即 grep 复核"，未再出现丢盘。
2. `base.py` 在本轮也被改动（`utc_now` 迁入），备份时未在你点名的 3 文件内——
   我按拆分前逐行读取的原文补了 `.bak_b19r2/base.py` 作为比对基线，已在报告中标注。
3. `item_convert.py` 初次覆盖率 27.9%、`document_models.py` 16.4%，未达项目 30% 标准，已补真实注释（字段设计动机/约束来源），非凑数。
