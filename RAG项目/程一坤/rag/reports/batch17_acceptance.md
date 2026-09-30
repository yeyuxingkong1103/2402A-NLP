# 批次 17 验收报告（17-1 健康自检 + 17-2 注释补齐）

日期：2026-09-21

## 17-1 scripts/check_services.py

用法：`python scripts/check_services.py`（默认不调外部 API）；`python scripts/check_services.py --with-api` 额外对 Embedding/Reranker/LLM 各做一次最小调用。

### 验收 a：正常运行（全绿）

```
=== 服务健康自检 ===
[1] Docker 容器
✅ 容器 my-redis            Up 9 minutes
✅ 容器 milvus-standalone   Up 8 minutes (healthy)
[2] Redis
✅ Redis ping（redis://127.0.0.1:6379/0）  PONG
[3] MySQL
✅ MySQL 连接（127.0.0.1:3306/legal_rag）
   documents=11  document_chunks=1627  law_versions=11  document_versions=11
[4] Milvus
✅ Milvus 连接（127.0.0.1:19530）
✅ 集合 legal_documents 存在
✅ 向量数一致性   Milvus=1627 == MySQL.document_chunks=1627
✅ 版本状态统计   approved=11   （approved=11，其它=0）
[5] 外部 API —— 跳过（加 --with-api 才会真实调用，避免浪费额度）
=== 结果 ===
✅ 全部通过（退出码 0）    EXIT=0
```

### 验收 b：docker stop my-redis 后

```
❌ 容器 my-redis   状态：Exited (0) Less than a second ago
   修复：docker start my-redis；若反复 Exited(134)，见《开发路线图》已知限制第 5 条
❌ Redis ping（redis://127.0.0.1:6379/0）
   连接失败：ConnectionError: Error 10061 ... 由于目标计算机积极拒绝，无法连接。
   修复：docker start my-redis
=== 结果 ===
❌ 2 项失败：my-redis, redis（退出码 1）    EXIT=1
```

### 验收 c：docker start my-redis 恢复后

```
=== 结果 ===
✅ 全部通过（退出码 0）    EXIT=0
```

### 实现说明

- 检查顺序与失败修复命令按要求：Docker（含 Exited(134) → 路线图已知限制第 5 条提示）→ Redis → MySQL（关键表行数）→ Milvus（连接/集合/向量一致性/版本状态统计）→ 外部 API（--with-api）。
- 向量数一致性口径：`get_collection_stats` 的 row_count 在当前 Milvus 版本上返回 0（统计缓存不可靠），已改用 `load_collection + query count(*)` 作权威口径（实测 1627 == document_chunks 1627）。
- 密钥安全：只输出 host:port 与库名；Redis URL 展示时剥离可能带密码的 userinfo；API 调用不回显 key。
- 退出码：全绿 0，任一失败 1（CI 可直接用）。

## 17-2 核心链路注释补齐

### 验收 a：行数与注释占比（改动前 → 改动后）

| 文件 | 行数 | 行内 # 注释 | docstring 行 |
|---|---|---|---|
| app/retrieval/assembly.py | 169 → 232 | 4 → 12 | 17 → 72 |
| app/retrieval/query_rewrite.py | 122 → 199 | 0 → 19 | 4 → 62 |
| app/memory/short_term.py | 54 → 124 | 0 → 6 | 10 → 74 |
| app/retrieval/context_builder.py | 169 → 197 | 44 → 52 | 27 → 47 |
| app/retrieval/fusion.py | 297 → 300 | 14 → 11 | 38 → 46 |

单文件均 ≤300 行（fusion.py 恰好 300）。

### 验收 b：去注释代码指纹（零逻辑改动证明）

自查命令（b17_fingerprint.py，AST 级）：
- 原理：注释/空行不进 AST；docstring 虽在 AST 中但属注释性质，故把模块/类/函数首个字符串表达式剥掉后对剩余结构做 `ast.dump` 哈希。前后一致 == 每个表达式、每个默认值完全一致。
- 命令：`python b17_fingerprint.py snapshot`（改动前）→ 改动 → `python b17_fingerprint.py check`（改动后比对）

```
文件                                             行数       #注释      doc行   指纹一致
backend/app/retrieval/assembly.py        169->232   4->12    17->72   ✅ 一致
backend/app/retrieval/query_rewrite.py   122->199   0->19     4->62   ✅ 一致
backend/app/memory/short_term.py          54->124   0->6     10->74   ✅ 一致
backend/app/retrieval/context_builder.py 169->197  44->52    27->47   ✅ 一致
backend/app/retrieval/fusion.py          297->300  14->11    38->46   ✅ 一致
结论： 全部零逻辑改动 ✅
```

### 验收 c：全量测试

```
413 passed, 5 warnings in 9.91s
```

### 验收 d：demo_ask 真实链路

```
【检索统计】向量召回 11 条 / 关键词召回 20 条 / 融合后 27 条 / 重排后 5 条
【引用法源】共 5 条：劳动合同法第47条、实施条例第27条、劳动合同法第97条、
            实施条例第25条、实施条例第10条
【回答】"经济补偿的核心算法就一句话：工作年限 × 月工资……[1][5][3][2][4]"
【护栏】citation_check_passed, guardrails_applied
```

如实说明：第一次跑时 LLM 单次输出未带 [n] 引用标记，触发护栏拒答兜底（拒答文案正常给出建议）；
复跑即给出带 5 处引用的完整回答。该波动与注释改动无关（指纹已证零逻辑改动），且恰证明护栏在真实工作。

### fusion.py 300 行处理说明（未拆文件的理由）

fusion.py 原本 297 行，按规范补注释必然超限。用户规则优先级为"优先精简注释，其次拆文件"。
考虑到拆文件会使"去注释代码指纹前后一致"的验收口径失效（文件结构改变），且本文件
RRF 融合与父块归并职责连续、拆分收益低，故采用：精简既有冗余注释（删 11 行低信息量
注释，含义并入更高质量的注释/docstring）+ 紧凑 docstring（Args/Returns 合行），
最终控制在 300 行整。未做任何逻辑改动（指纹一致）。

## 边界自查

- docs/ 未动 ✅（17-1 的说明写在本报告，未写入 docs）
- 零新增依赖 ✅（脚本只用 pymysql/pymilvus/redis 等 backend 既有依赖）
- 临时脚本不进 backend/ ✅（b17_fingerprint.py 在项目根）
- 未改动既有接口/函数签名/逻辑 ✅（AST 指纹一致）
