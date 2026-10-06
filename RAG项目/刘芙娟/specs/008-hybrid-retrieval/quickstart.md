# Quickstart / 验收指南: 混合检索（S9）

**Feature**: `008-hybrid-retrieval` | **Date**: 2026-09-28

本文件给出**可复制的命令**与**预期结果**，用于验证本特性端到端可用。每条命令都对应 spec.md 里的一条成功标准（SC）。

> ⚠️ 本文件不含实现代码。实现细节见 `plan.md` 与 `contracts/`。

---

## 0. 前置条件（**必须先满足，否则后面每一步都会失败**）

### 0.1 Milvus 必须已启动 ← **实测当前未运行**

```bash
docker ps --format "{{.Image}}" | grep milvus
# 预期：milvusdb/milvus:v2.6.9
```

若没有输出，启动它：

```bash
docker start milvus-standalone
```

> **这是本特性带来的行为变更**：`backend/serve.py` 以前在 Milvus 未运行时也能启动（指纹门禁只读 `index_manifest.json`）。接入混合检索后，**Milvus 成为启动的硬前提** —— 关键词索引在启动期从库里拉语料（FR-010），拉不到就不该启动。
>
> 若容器名不同：`docker ps -a --format "{{.Names}}\t{{.Image}}"` 查实际名字。

### 0.2 `.env` 必须存在且检索组配置齐全 ← **实测当前不存在**

```bash
cp .env.example .env
```

然后**至少**填这几项（`backend/api/config.py` 已把它们转为必需项）：

| 变量 | 填什么 | 依据 |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | S6 实测 |
| `MILVUS_COLLECTION` | `med_rag_v1` | `backend/index/__init__.py` |
| `MILVUS_TOKEN` | 留空 | S6 实测服务端未启用认证 |
| `EMBED_MODEL_PATH` | `E:\资料\BAAI--bge-m3` | S5 / `backend/embed/__init__.py` |
| `SIMILARITY_THRESHOLD` | **先填 `0.6` 再按 §3 标定** | constitution `TODO(SIMILARITY_THRESHOLD)` 未标定 |
| `TOP_K` | `3` | `docs/05` §8 裁决 |

**新增的可选项**（有默认值，可先不填）：

```ini
RETRIEVAL_CANDIDATES=20    # 每路取多少候选再融合
RRF_K=60                   # RRF 平滑常数
LEXICAL_ADMIT_RANK=3       # 关键词名次 ≤ 此值可绕过语义阈值
BM25_K1=1.5
BM25_B=0.75
```

### 0.3 知识库非空

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -c "import json;d=json.load(open('data/index_manifest.json',encoding='utf-8'));print(d['total_chunks'],'chunks /',len(d['documents']),'docs')"
# 预期：65 chunks / 1 docs
```

若向量库是空的，先跑 S6 入库（`backend/index_milvus.py`，见 `specs/005-milvus-ingest/quickstart.md`）。

---

## 1. 纯函数自检（**不连 Milvus、不加载权重，5 秒内出结果**）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search selfcheck
```

**预期**：9 条断言全部 PASS，退出码 0。

**这一条先跑的理由**：BM25 打分与 RRF 融合是本特性唯一"算错了也不会报错"的部分（排名只是变得不那么好，没有异常）。先把它们钉死，再连真实数据 —— 否则数据出了问题，分不清是检索逻辑错还是语料错。

**覆盖**：见 `contracts/cli.md` §2 的 SC-1 ~ SC-9。

---

## 2. 语料自检（**连 Milvus，验证两路共用同一份语料**）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search corpus
```

**预期**：

```
语料条数: 65
manifest total_chunks: 65           [一致]
去重后 chunk_id: 65                 [一致]
text 为空: 0
file_name 为空: 0
page_start > page_end: 0
平均文档长度(tokens): <某个 100–400 之间的数>
```

**退出码 0**。条数与 manifest 不符则退出码 2 —— 那说明库被改过而 manifest 未更新，**在往下走之前必须先查清是哪一边过时了**（对应 R3）。

---

## 3. 单条检索与阈值标定（**constitution 的 `TODO(SIMILARITY_THRESHOLD)` 在这里回填**）

### 3.1 语义类问题（对应 US2 / SC-002）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "血压高平时要注意什么"
```

**预期**：
- `语义路` 命中数 > 0，首条余弦明显高于其他条
- 最终判定的 `is_empty=False`
- 输出的前 3 条里，至少 1 条的人工判断是"确实相关"

### 3.2 关键词类问题（对应 US1 / SC-001 —— **本特性的核心价值验证**）

先找一个**确认出现在原文里**的术语：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search corpus | tail -20
# 从"高频词 top10"里挑一个医疗术语；或直接翻开 data/chunks/*.chunks.jsonl 找一个药名/编号
```

再用它作为问题检索：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "<那个术语>"
```

**预期 —— 这是 SC-001 的验收点**：
- `关键词路` 首条 BM25 分明显高于其他条
- **最终第 1 条引用的 `text` 中包含该术语**
- 该条 `余弦` 可能不高（这正是要观察的现象）
- 若该条余弦 < `SIMILARITY_THRESHOLD`，则判定应为 `below_threshold=True`（全部靠关键词纳入）—— **这是正确行为，不是故障**（R7）

### 3.3 混合是否真的比单路好（可测量，不靠感觉）

```bash
# 纯关键词路的排名
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "<术语>" --no-semantic > /tmp/lex.txt

# 纯语义路的排名（复现 S9 之前的基线）
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "<术语>" --no-lexical > /tmp/sem.txt

# 混合（默认）
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "<术语>" > /tmp/hyb.txt
```

对比三份输出：**首次命中该术语的排名，混合应不劣于任何单路**。若混合比某一路差，说明融合规则有问题 —— 这是本特性唯一需要"调参"的地方，而 `RRF_K` 是平滑常数、通常不该动（R5），要动先怀疑候选集大小 `RETRIEVAL_CANDIDATES`。

### 3.4 标定阈值

对 5–10 条**已知正确答案**的问题各跑一次，记录命中条目的余弦值：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "<问题>" --json | \
  D:/zg6_Project/9/med_rag/rag/python.exe -c "import sys,json; d=json.load(sys.stdin); [print(f\"{p['score']:.4f}  {p['chunk_id']}\") for p in d['passages']]"
```

取"正确答案的余弦"与"错误答案的余弦"之间的分界作为 `SIMILARITY_THRESHOLD`，**回填三处**：
1. `.env`
2. `.env.example`
3. `.specify/memory/constitution.md` 的 `TODO(SIMILARITY_THRESHOLD)` 与 `docs/02_架构图.md` §10

### 3.5 无关问题应被拒答（对应 SC-004）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "今天晚饭吃什么好"
```

**预期**：`is_empty=True`。

⚠️ **这一条曾经不通过，修法与原因值得记住**：只卡 `LEXICAL_ADMIT_RANK` 时，
BM25 对这个提问照样返回前 3 名（`晚饭`/`好` 在这份语料里都很罕见），于是被判成
"命中"。修法是给关键词准入加**查询词覆盖率**门槛（`LEXICAL_MIN_COVERAGE`，默认
0.5）—— 该提问 top-3 覆盖率仅 0.20，而术语照抄是 1.00。

CLI 表格里的 `覆盖率` 一列就是标定这一项的依据：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "今天晚饭吃什么好"
# 看 top-3 的「覆盖率」列 —— 应显著低于 0.50
```

若它仍 ≥ 0.5，说明这个提问确实与该语料有词面重叠（BM25 分不开），此时
**正确的处置是补阈值或补语料，不是继续调覆盖率** —— 覆盖率只该用来挡"查询词
在库里根本不存在"的闲聊式提问。

---

## 4. 可复现性（对应 SC-006）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "血压高平时要注意什么" --json > /tmp/run1.json
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "血压高平时要注意什么" --json > /tmp/run2.json
diff /tmp/run1.json /tmp/run2.json && echo "SC-006 PASS"
```

**预期**：无差异。有差异说明排序不稳定（检查 `fuse.py` 的排序键是否带 `chunk_id` 末位，INV-3）。

---

## 5. 接入服务端（对应 SC-003 / SC-008）

### 5.1 启动服务

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.serve > .smoke_out/serve.log 2>&1
```

**预期启动输出**（顺序固定；下列为 2026-09-28 实测，耗时是本机数据）：

```
正在加载 BGE-M3 权重（约 2.3 GB，首次约 10 秒）…
权重已加载（6.9 s），编码指纹 26159e7a…
指纹与索引一致（med_rag_v1 / 65 chunks）
正在构建关键词索引（从向量库拉取语料）…            ← 新增
关键词索引就绪（65 条，与索引清单一致，1.3 s）      ← 新增（SC-009：语料为空会在这一步暴露）
检索配置：top_k=3 threshold=0.6000 candidates=20 rrf_k=60 admit_rank=3 min_coverage=0.50
医知源服务已启动  http://127.0.0.1:8000/
```

**若 Milvus 未启动**，应在第 ④ 步失败并以非 0 退出，提示里明确指出 `MILVUS_URI` 与 `docker ps`。**MUST NOT** 出现"启动成功但检索永远为空"。

### 5.2 浏览器验证（SC-003 / SC-008）

打开 `http://127.0.0.1:8000/`，提交 §3.1 与 §3.2 里的问题。

**预期**：
- 提交后 **≤ 2 秒**引用区出现卡片（SC-003）
- 每张卡片显示：序号、**文件名**、**页码**、**分数**、原文预览
- 原文超过预览长度的卡片有「展开原文」按钮，点击后就地展开
- 正文区显示 `RETRIEVAL_READY_NOTICE` 的过渡文案（命中时）
- 提交无关问题时，引用区为空态、正文区显示拒答兜底话术

**SC-008 的硬验证**：

```bash
# 本特性实现前后，前端文件 MUST 逐字节相同
git diff --stat frontend/          # 预期：无输出
# 若仓库未纳入版本管理，比较实现前后的 mtime 或做一次 hash 比对
```

若前端确实需要改动才能正确显示，说明**服务端产出的 `citation` 形状偏离了 `docs/05` §3.1.3** —— MUST 改服务端（FR-018）。

### 5.3 服务端日志检查（对应 FR-019 / FR-020）

```bash
grep "检索完成" .smoke_out/serve.log | tail -5
```

**预期**：每次提问一条，形如

```
检索完成 answer_id=<uuid> is_empty=False below_threshold=False semantic=20 lexical=20 fused=21 returned=3 top_cosine=0.7123 top_rrf=0.03252 elapsed_ms=42 state=ok
```

**必须能区分**（FR-020）：`semantic=-1` / `lexical=-1` 表示**该路执行失败**，`0` 表示无命中。看到 `-1` 说明有一路坏了，而不是知识库里没有。

---

## 6. 边界与失败路径

| 场景 | 操作 | 预期 |
|---|---|---|
| 空问题 | 前端提交空白 | 前端拦截，**不发请求**（S7 既有行为） |
| 超长问题 | 提交 > 200 字符 | 前端提示，不发请求（S7 既有行为） |
| 纯标点问题 | `search "？？？"` | 关键词路 0 命中，语义路正常返回；`search` 不抛异常 |
| Milvus 启动后挂掉 | 服务运行中 `docker stop milvus-standalone`，再提问 | ERROR 日志 `retrieval_failed`，用户看到拒答兜底话术，**不是 500** |
| 库为空 | 换一个空 collection 名启动 | 启动期失败，提示需先运行 S6 入库 |
| 语料与 manifest 不符 | 手工改 `index_manifest.json` 的 `total_chunks` | 启动时**告警但不退出**（R3：库可能刚被重跑） |

---

## 7. 前后端联调自检清单

- [ ] `selfcheck` 9 条断言全 PASS
- [ ] `corpus` 的条数与 `index_manifest.json` 一致
- [ ] 术语类问题的**首条引用原文包含该术语**（SC-001）
- [ ] 口语化问题引用区非空（SC-002）
- [ ] 无关问题走拒答（SC-004）
- [ ] 同问题两次 `--json` 输出逐字节相同（SC-006）
- [ ] `git diff --stat frontend/` 无输出（SC-008）
- [ ] 浏览器里引用卡片有序号/文件名/页码/分数/预览/展开（SC-008）
- [ ] 提交到看见引用的等待 ≤ 2 s（SC-003）
- [ ] 一次性检索日志六项齐全（FR-019）
- [ ] `find backend/retrieve -name "*.py" | xargs wc -l` 每行 ≤ 300（SC-007）
- [ ] `SIMILARITY_THRESHOLD` 已回填 `.env` / `.env.example` / constitution / `docs/02` §10

---

## 7.1 实测记录（2026-09-28）

以下是实现完成后在真实库（`med_rag_v1`，65 chunks）上跑出的结果，供后续回归对照。
命令用 `MILVUS_URI=… EMBED_MODEL_PATH=… SIMILARITY_THRESHOLD=0.6 TOP_K=3` 从进程
环境注入（当时仓库尚无 `.env`）。

| 用例 | 查询 | `is_empty` | `below_threshold` | 返回引用 | 终帧一致 | SC |
|---|---|---|---|---|---|---|
| US1 术语照抄 | `氢氯噻嗪` | False | True | 3（首条含该术语） | ✓ | SC-001 |
| US1 术语照抄 | `高血压` | False | False | 3 | ✓ | SC-001 |
| US2 语义 | `血压高平时要注意什么` | False | False | 3（top 余弦 0.7182） | ✓ | SC-002 |
| US2 语义 | `阿司匹林怎么吃` | False | False | 1（余弦 0.6467，正是写着剂量的那一段） | ✓ | SC-002 |
| SC-004 无关 | `今天晚饭吃什么好` | True | True | 0 | ✓ | SC-004 |
| SC-004 无关 | `介绍一下量子力学` | True | True | 0 | ✓ | SC-004 |

- `自检通过（10 项，0 项失败）`，退出码 0
- `corpus` 自检通过：65 条、与清单一致、字段无缺失、平均文档长度 165.7 tokens
- 检索耗时 0–47 ms（SC-003 的 2 s 预算几乎全部属于 S8 的编码）
- 服务启动耗时：权重 6.9 s + 关键词索引 1.3 s
- 前端 7 个文件的 mtime 全部停留在 2026-09-27，**零改动**（SC-008）

**实现期发现并修掉的两个真实缺陷**（都不是靠读代码看出来的，是靠跑真实数据）：

1. **无关提问被放行** —— 关键词准入只卡名次时，`今天晚饭吃什么好` 被判为命中。
   修法：加查询词覆盖率门槛（FR-013a）。
2. **终帧清空引用区** —— `done` 事件里 `citations` 恒为空数组，而前端 `renderDone`
   用它重渲染。表现为用户看到的引用在终帧到达那一刻消失。修法：终帧复用
   `citations` 事件的那份 payload。

---

## 8. 已知不在本特性的范围

- **答案生成**：I-06 未实现，命中后正文是过渡文案（D3），不是真答案。
- **紧急症状前置话术**：I-04 未实现；`stream.py` 的位置结构保留但不触发。
- **`/health` 的 `matches_index` 检查**：`docs/05` §3.3 定义但未实现；因此「重新入库后 MUST 重启服务」这条前提仍然成立。
- **检索性能优化**：当前规模（65 chunks）下无需优化；`R12` 记录了何时必须改（并发 > 1 或检索耗时 > 100 ms）。
- **`pymilvus` 版本声明的核实**：声明 `>=2.6,<3` vs 实际 `3.0.2`。这是实现期的**首个任务**（见 `tasks.md`），不属于验收范围。
