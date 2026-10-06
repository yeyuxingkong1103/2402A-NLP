# 契约：检索命令行（`backend/retrieve_search.py`）

**Feature**: `008-hybrid-retrieval` | **Version**: 1.0 | **Date**: 2026-09-28

本文件定义检索脚本的命令行接口。**CLI 与运行时服务 MUST 共用同一份实现**（FR-023）—— CLI 只做参数解析与输出渲染，检索逻辑全部来自 `backend/retrieve/service.py`。

---

## 1. 调用方式

**MUST 用模块形式，以仓库根为工作目录**（与既有的 `backend/serve.py`、`backend.index_milvus` 同一约定）：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search <子命令> [选项]
```

> ⚠️ MUST NOT 写 `rag/python.exe backend/retrieve_search.py` —— 那样 `sys.path[0]` 会变成 `backend/` 而不是仓库根，`import backend.retrieve` 会失败。此约定在 `backend/serve.py` 的模块文档字符串里已有先例。

**为什么必须有 CLI**：constitution 的 `TODO(SIMILARITY_THRESHOLD)` 至今未回填 —— 阈值从未标定过。标定要做的事是"对若干条已知答案的问题反复试跑、看两路分数分布"，这个过程 MUST 能脱离 Web 服务与浏览器完成。CLI 不是服务端点的复制品，它是**标定工具**。

---

## 2. 子命令

### `search` —— 对单条问题执行检索

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search search "血压高平时要注意什么"
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--top-k` | 取配置值 | 覆盖配置里的 `TOP_K` |
| `--threshold` | 取配置值 | 覆盖配置里的 `SIMILARITY_THRESHOLD`（**标定的主要旋钮**） |
| `--candidates` | 取配置值 | 覆盖 `RETRIEVAL_CANDIDATES` |
| `--json` | 关 | 以 JSON 输出 `RetrievalResult`，供脚本消费 |
| `--no-semantic` | 关 | **仅标定用**：跳过语义路，只看关键词路（见 §4） |
| `--no-lexical` | 关 | **仅标定用**：跳过关键词路，复现 S9 之前的纯语义基线 |

`--no-semantic` / `--no-lexical` 是**标定工具**，不是降级开关：它们让"混合比单路好多少"这件事可测量。二者同时给出时 MUST 报参数错误退出。

**输出**（默认人类可读，逐条）：

```
问题: 血压高平时要注意什么
向量: 已计算（1024 维，指纹 26159e7a…）
语料: 65 chunks（与 index_manifest.json 的 total_chunks 一致）

语义路  命中 20 条   首条余弦 0.7123
关键词路 命中 20 条   首条 BM25 8.4210
融合后  21 条（去重 19 条）

#  排名  chunk_id             余弦    BM25  语义名次 关键词名次  RRF     来源
1  1     d6da41b5d356:0042    0.7123  6.104    1        3       0.03252  两路
2  2     d6da41b5d356:0017    0.6891  0.000    2        -       0.01613  语义
3  3     d6da41b5d356:0055    0.2310  8.421    -        1       0.01639  关键词

判定: is_empty=False  below_threshold=True（全部靠关键词纳入）
耗时: 42 ms
```

**必须显示的列**：`余弦`、`BM25`、`语义名次`、`关键词名次`、`RRF`、`来源`。
**为什么必须全显示**：融合规则的有效性只能靠回看两路各自的原始名次来验证（data-model.md §2.1）。只显示最终排名，CLI 就退化成一个"看起来对不对"的工具，标定不了任何参数。

**退出码**：0 = 成功（含 `is_empty=True`；"检索为空"是正常结论，不是失败）。

---

### `selfcheck` —— 纯函数自检

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search selfcheck
```

**不连 Milvus、不加载模型权重**，只对 `fuse.py` 与 `lexical.py` 的纯函数跑断言。这是本项目**没有 pytest** 的替代方案（R11）。

覆盖的断言：

| 编号 | 断言 | 对应不变量 |
|---|---|---|
| SC-1 | BM25 IDF 恒正；`df = n_docs` 时 IDF `> 0` | R1 |
| SC-2 | BM25 分数对词频单调不减（`k1` 饱和但不反转） | R1 |
| SC-3 | 空语料下 BM25 返回空列表且不抛异常 | Edge Case |
| SC-4 | RRF：两路都命中的片段得分 > 仅单路命中 | R5 |
| SC-5 | RRF：`k` 增大时头部名次差距缩小 | R5 |
| SC-6 | 排序键在 RRF 同分时按 `(-cosine, chunk_id)` 稳定（INV-3） | SC-006 |
| SC-7 | 同一 chunk 在两路都出现时，融合结果只保留一条且合并两个名次 | FR-005 |
| SC-8 | `below_threshold` 四象限取值与 data-model.md §1.2 一致（INV-5/6/7） | FR-013 |
| SC-9 | `tokenize()` 满足：同一文本两次调用结果相同；索引期与查询期走同一函数 | R2 |

**退出码**：0 = 全部通过；2 = 有断言失败（逐条打印失败项）。

---

### `corpus` —— 语料自检

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.retrieve_search corpus
```

连 Milvus 拉语料并报告：

```
语料条数: 65
manifest total_chunks: 65           [一致]
去重后 chunk_id: 65                 [一致]
text 为空: 0
file_name 为空: 0
page_start > page_end: 0
平均文档长度(tokens): 187.3
高频词 top10: …
```

**必须检查的三件事**（任一不符即非 0 退出）：
1. 拉回条数 == `index_manifest.json` 的 `total_chunks` —— 不符说明库被改过而 manifest 未更新（R3）。
2. 拉回条数 < 请求的 `limit` —— 若相等则**告警可能被截断**（R3，与 `backend/index/store.py` 的 `QUERY_LIMIT` 同一理由）。
3. `chunk_id` 唯一 —— 不唯一会让"去重"这一步失去意义。

**退出码**：0 = 一致；2 = 与 manifest 不符；3 = Milvus 不可达。

---

## 3. 退出码

沿用 `docs/05` §5 的分工（1 参数 / 2 校验 / 3 外部依赖）：

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 参数错误（未知子命令、互斥选项同时给出、数值越界） |
| 2 | 数据/校验问题（自检失败、语料与 manifest 不符） |
| 3 | 外部依赖不可用（Milvus 不可达、模型缺失） |

与既有 `backend/index/__init__.py` 的退出码语义一致。

---

## 4. CLI MUST NOT 做的事

| 禁止 | 理由 |
|---|---|
| 自己实现检索 | FR-023：CLI 与运行时 MUST 共用同一份实现 |
| 写 Milvus | 本特性是纯读的（结构约束 9） |
| 打印查询向量数值 | 与日志契约同一理由（检索日志 MUST NOT 记录 `query_vector`） |
| 把 `--no-semantic` / `--no-lexical` 暴露给服务端 | 它们是标定工具。服务端只跑完整混合检索 —— 单向降级正是 spec Edge Case 禁止的静默失效 |
| 改 `chunk.text` 的展示（截断、转义、加省略号） | CLI 的用途之一是核对命中是否真的是原文。截断会让"命中了但看着不对"变成无法判断 |

> 最后一条的例外：`--json` 输出完整不截断；人类可读输出的 `text` 打印**全文**。若终端太长，是终端的问题，不是渲染层应该替用户决定的事 —— 这一点与前端 `transcript.js` 的"预览 + 展开"不同：界面上同时有几十条，终端上一次只有几条。
