# 批次 19 第 1 轮验收报告：拆分严重超标文件（3 个）

日期：2026-09-21

## 新增文件登记（新增文件 → 职责一句话）

### ① app/chat/service.py（516 → 217 行）
| 新文件 | 行数 | 职责 |
|---|---|---|
| app/chat/result.py | 42 | ChatResult 结果类型与法条摘录组装（[n] 与提示词编号一致） |
| app/chat/memory_hooks.py | 101 | 长期记忆读写钩子 Mixin（检索前读、回答后后台线程写，失败不影响主流程） |
| app/chat/streaming.py | 157 | 流式问答主流程 chat_stream（真流式 + replace 事件护栏衔接） |
| app/chat/bootstrap.py | 64 | build_default_chat_service 默认依赖装配（检索/LLM/记忆实例组装） |
| app/chat/service.py（瘦身后） | 217 | 同步问答主流程编排（chat + __init__），继承两个 Mixin |

调用方同步：api/chat.py、cli/demo_ask.py（build_default_chat_service → app.chat.bootstrap）；
测试 4 处 ChatResult import → app.chat.result（test_chat_service / test_cli_contract /
test_chat_api ×2 / test_chat_sessions_api）。无兼容垫片。

### ② app/memory/long_term.py（391 → 232 行）
| 新文件 | 行数 | 职责 |
|---|---|---|
| app/memory/memory_schema.py | 94 | Milvus 集合管理：独立集合常量、MemoryRecord、异常类型、建集合（幂等）与字段清单 |
| app/memory/memory_summarize.py | 49 | 写入前的摘要生成（LLM 把一次问答提炼成用户事实，失败返回 None） |
| app/memory/memory_settings.py | 47 | 用户级记忆开关（MySQL users 表，接口 8.3） |
| app/memory/long_term.py（瘦身后） | 232 | 记忆读写业务：写入去重、检索、分页列表、软删除 |

调用方同步：api/memory.py、chat/bootstrap.py、tests（test_memory_api /
test_long_term_memory）。无兼容垫片。

### ③ app/ingest/chunker.py（383 → 181 行）
| 新文件 | 行数 | 职责 |
|---|---|---|
| app/ingest/article_splitter.py | 60 | 条（一级）边界切分："第 X 条"正则识别条文列表 |
| app/ingest/paragraph_items.py | 89 | 款/项识别：条内自然段（款）与「（一）（二）」项切分、中文项号转阿拉伯数字 |
| app/ingest/chunk_text.py | 50 | 文本工具：归一化/检索文本拼接/文档标识（避免 chunker⇄paragraph_items 循环导入） |
| app/ingest/chunk_fingerprint.py | 45 | 切块指纹与版本号（CHUNKER_VERSION，增量判定用） |
| app/ingest/chunker.py（瘦身后） | 181 | DocumentChunk 数据结构 + chunk_document 组装主流程 |

调用方同步：ingest/offline_ingest.py（compute_chunk_fingerprint → chunk_fingerprint）、
tests/test_chunker_paragraphs.py（CHUNKER_VERSION → chunk_fingerprint）。无兼容垫片。

## 验收

### AST 逻辑零改动（剥 docstring + import 后逐对象比对）
说明：拆分前未留备份（失误），基线从本轮逐行读取的原文重建为 b19_orig_*.py；
比对键为函数裸名（允许方法换宿主类，如 ChatService→Mixin）。
```
=== chat/service.py ===
函数: 原 8 个 | 缺失 [] | 不一致 [] | 新增逻辑 []
类字段: 原 2 个 | 缺失 [] | 不一致 []
顶层常量: 缺失 0 | 新增 0
PASS

=== memory/long_term.py ===
函数: 原 14 个 | 缺失 [] | 不一致 [] | 新增逻辑 []
类字段: 原 4 个 | 缺失 [] | 不一致 []
顶层常量: 缺失 0 | 新增 0
PASS

=== ingest/chunker.py ===
函数: 原 8 个 | 缺失 [] | 不一致 [] | 新增逻辑 []
类字段: 原 1 个 | 缺失 [] | 不一致 []
顶层常量: 缺失 0 | 新增 0
PASS

OVERALL: PASS
```

### 行数（新文件全部 ≤250）
```
service 217 / result 42 / memory_hooks 101 / streaming 157 / bootstrap 64
long_term 232 / memory_schema 94 / memory_summarize 49 / memory_settings 47
chunker 181 / chunk_text 50 / article_splitter 60 / paragraph_items 89 / chunk_fingerprint 45
```

### 全量测试
```
413 passed, 5 warnings in 11.42s
```

### demo_ask "经济补偿怎么算"
```
向量召回 11 / 关键词召回 20 / 融合后 27 / 重排后 5
引用 5 条（劳动合同法 47/97 条、实施条例 27/25/10 条）
【护栏】citation_check_passed, guardrails_applied
```

## 全 app/ 超标名单（>300 行，剩余 6 个，即第 2/3 轮目标）
```
   327 app/retrieval/service.py      （第 2 轮）
   324 app/db/sql_models.py          （第 2 轮）
   318 app/db/import_service.py      （第 2 轮）
   309 app/ingest/law_metadata_extractor.py（第 3 轮）
   308 app/ingest/offline_ingest.py  （第 3 轮，拆分后 +1 行）
   301 app/review/review_service.py  （第 3 轮）
```

## 其他
- b18 临时脚本（b18_fusion_original.py、b18_verify.py）已删。
- 本轮临时文件保留在项目根：b19_orig_*.py（3 份基线）、b19_verify.py（第 2/3 轮复用），
  全部完成验收后可删。
- 教训记录：拆分前必须先备份原文件（本轮基线是从上下文重建的，费时且有风险）；
  AST 比对脚本已修正三处（try 块内 import 剥离、同名方法多重集比对、确定性清洗），
  第 2/3 轮直接复用。
- 零新增依赖；docs/、data/、tests 内容未动（tests 仅 import 行同步更新）。
