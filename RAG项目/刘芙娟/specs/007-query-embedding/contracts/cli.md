# Contract: CLI（`backend/query_embed.py`）

**Feature**: `007-query-embedding` | **Date**: 2026-09-27

本特性唯一的进程外接口。**服务端与它共用同一份编码实现**（FR-012/FR-032），因此这里不存在"CLI 的口径"与"服务的口径"之分。

```text
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.query_embed <子命令> [选项]
```

**必须用模块形式（`-m`）**，且工作目录为仓库根。直接 `python backend/query_embed.py` 会让 `sys.path[0]` 变成 `backend/`，`import backend.embed` 失败。既有管线的 `-m backend.pipeline` 约定同理（`docs/05` §5）。

---

## 1. 子命令

| 子命令 | 作用 | 读 | 写 |
|---|---|---|---|
| `embed` | 批量向量化**尚未完成**的记录 | `data/questions/*.jsonl` | 回写同一批文件 |
| `verify` | 只读自检：门禁 + 格式 + 向量范数 | 上述 + `index_manifest.json` | **不写** |
| `purge` | 按日期清理留存 | — | 删除 `data/questions/{date}.jsonl` |
| `show` | 打印某条记录的可读摘要 | `data/questions/*.jsonl` | **不写** |

### `embed`

```text
-m backend.query_embed embed [--date YYYYMMDD] [--dry-run] [--limit N]
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--date` | 全部日期 | 只处理某一天的文件 |
| `--dry-run` | 关 | 只报告将处理多少条，不写 |
| `--limit` | 不限 | 最多处理 N 条（调试用） |

**幂等性（FR-013）**：只处理 `status != "ok"` 的记录。对同一批文件重复运行，第二次处理的条数 MUST 为 0（SC-009）。

**回写方式**：读入整个文件 → 替换需处理的行 → **原子替换**（写 `.tmp` 再 `os.replace`，与 `backend/embed/store.py` 既有模式一致）。追加写只用于**新增记录**，批量回写必须整文件原子替换 —— 否则中途失败会留下半个文件。

### `verify`

```text
-m backend.query_embed verify
```

只读自检，输出四项结论：

1. 门禁：查询侧指纹 vs `index_manifest.json` 的 `embed` 块，逐字段列出差异（无差异则列出 10 项「一致」）；
2. 格式：每一行都是合法 JSON、字段齐全、字段顺序固定；
3. 向量：`status == "ok"` 的行，向量维度为 1024、L2 范数与 1 的偏差 < 1e-5；
4. 统计：总条数 / `ok` / `failed` / 未处理（若有）。

**退出码非 0 表示自检未通过**，可用于自动化。

### `purge`

```text
-m backend.query_embed purge --before YYYYMMDD [--dry-run]
-m backend.query_embed purge --date YYYYMMDD
```

**默认 `--dry-run`**，需显式 `--yes` 才真删。这是破坏性操作（FR-028 的清理入口），与既有管线的 `--confirm-rebuild` 同一取向（`docs/05` §5：破坏性操作必须由人显式发起）。

删除前 MUST 打印：将删除的文件名、条数、字节数。

### `show`

```text
-m backend.query_embed show --answer-id <uuid>
```

打印该记录的可读摘要：问题、时间、状态、向量前 5 维与范数、编码指纹的 `rule_version`。**向量本身不相容地打印**（1024 个数字对排障无用），只给摘要。

---

## 2. 退出码

| 码 | 含义 | 触发场景 |
|---|---|---|
| 0 | 成功 | — |
| 1 | 参数错误 | 子命令缺失、选项冲突、日期格式错 |
| 2 | 门禁失败 | 指纹不一致、清单缺失或无法解析（FR-016/FR-018） |
| 3 | 模型不可用 | 目录不存在、权重缺失或残缺、加载失败 |
| 4 | 数据异常 | JSONL 行损坏、字段缺失、向量维度/范数不符 |

**沿用 `docs/05` §5 的分工**（1 参数 / 2 校验 / 3 外部依赖），4 为 `backend/index/` 已有的「数据问题」类别，本特性沿用同一语义。

**退出码 2 与 3 的区分很重要**：2 是"你的索引和你的模型口径对不上"（去查 `index_manifest.json`），3 是"模型本身有问题"（去查权重文件）。合在一起会让排障从"看一眼"变成"逐项试"。

---

## 3. 输出约定

- **人读的部分走 stdout**，用中文，带分隔线，与 `backend/embed_chunks.py` 的报告风格一致；
- **机器读的部分不引入**（不输出 JSON）——本特性没有自动化消费者，加一个 `--json` 是为不存在的需求付复杂度；
- 错误走 stderr，且 MUST 是**可读的完整句子**，MUST NOT 只打印异常类型；
- 进度：批量处理时每批打印一行（与 `embed_chunks.py` 的 `on_batch` 回调一致）。

---

## 4. 与既有 CLI 的关系

`docs/05` §5 把数据管线 CLI 定义为 `python -m backend.pipeline <子命令>`。**本特性的 CLI 不并入其中**，理由：

| | `backend.pipeline`（S2–S6） | `backend.query_embed`（S8） |
|---|---|---|
| 处理对象 | **语料**（文档 → chunk → 向量 → 索引） | **用户问题** |
| 调用者 | 运维，低频、离线 | 运维（批量/清理），且服务端**共用同一实现** |
| 破坏性 | 有（`rebuild`） | 有（`purge`） |
| 与运行时的关系 | `docs/05` §5 明确「不被后端进程调用」 | **被后端进程调用**（同一份 `service.py`） |

最后一行是决定性的：把它并进 `backend.pipeline` 会让"离线管线不被后端进程调用"这条边界失效，而那条边界是 `docs/05` §5 用来保证"改分块参数不会碰到正在服务的库"的。
