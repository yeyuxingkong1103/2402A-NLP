# Phase 1 Contract: 命令行接口

**Feature**: `005-milvus-ingest` | **Date**: 2026-09-27

---

## 1. 调用形式

```text
D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py [doc_ids ...] [选项]
```

**唯一入口**。`doc_ids` 为位置参数，**留空 = 处理 `data/chunks/` 下全部文档**（D3，沿用
`backend/embed_chunks.py:219` 的既有约定）。

## 2. 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--uri <url>` | `http://localhost:19530` | Milvus 地址 |
| `--chunks-dir <path>` | `<repo>/data/chunks` | 分块产物目录（输入） |
| `--emb-dir <path>` | `<repo>/data/embeddings` | 向量产物目录（输入） |
| `--manifest <path>` | `<repo>/data/index_manifest.json` | 索引清单（产出） |
| `--backup-dir <path>` | `<repo>/data/index_backup` | 回滚备份目录（产出） |

**没有 `--collection` 开关**（实现阶段的决定，理由如下）：collection 名是 **V1 锁定**的契约 ——
`docs/04 §9.1` 定 schema、`§9.3` 的清单里硬编码 `"collection": "med_rag_v1"`，将来的后端也按这个
名字读。让它可配只会制造一种错误：把数据写进后端不会读的 collection，而脚本报告"成功"。
MUST NOT 提供这种开关。

**输入与产出都在项目内的 `med_rag\data\` 下**，与 `docs/04 §9.3` 的 `data/index_manifest.json`
一致。目录布局：

```text
med_rag\data\
├── chunks\            输入（S4）
├── embeddings\        输入（S5）
├── parsed\  clean\  source_data\   输入（S2/S3/S1）
├── index_manifest.json            产出（S6）
└── index_backup\                  产出（S6，回滚依据）
```
| `--create-collection` / `--no-create-collection` | 创建 | collection 不存在时是否自动创建（D3 定为创建；提供反面开关便于"只许写已有库"的场景） |
| `--rebuild` | 否 | **破坏性**：忽略门禁，按 `doc_ids`（或全部）重建。MUST 显式给出才生效（FR-020） |
| `--dry-run` | 否 | 只做 V1–V12 校验与门禁判断，**不写库、不写 manifest**。退出码语义不变 |
| `--print-config` | 否 | 打印本次的 `pipeline_config` 与 `pipeline_config_hash` 后退出（不连库） |
| `--print-env` | 否 | 打印运行环境（解释器版本、`pymilvus` 版本、`numpy` 版本、连接地址、collection 名）后退出 |

**凭据**：只从环境变量 `MILVUS_TOKEN` 读（宪法原则 III）。未设置 → 按无凭据连接（该实例实测无认证）。**无命令行参数接受凭据**，避免它出现在 shell 历史与进程列表里。

## 3. 退出码

`0`–`3` 沿用 `docs/05 §5`；`4`/`5` 为本步骤新增，**不与既有含义冲突**。

| 码 | 含义 | 库的状态 |
|---|---|---|
| `0` | 成功 | 全部目标文档已提交 |
| `1` | 参数错误（如 `--rebuild` 与 `--dry-run` 同时给出、`doc_ids` 指向不存在的产物） | 未触碰 |
| `2` | 校验失败（V1–V12 任一不过）；**或**写入后校验失败但**已成功回滚** | 未触碰 / 回到运行前 |
| `3` | 外部依赖不可用（Milvus 连不上、`pymilvus` 未安装、服务端版本不支持 FLAT/COSINE） | 未触碰 |
| `4` | **版本门禁拒绝**：库内 `pipeline_config_hash` 与本次不符，需人工决定是否 `--rebuild` | 未触碰 |
| `5` | **写入后校验失败，且回滚也失败**：库状态不确定，**必须人工介入** | **不确定**；错误信息给出备份文件路径 |

多份文档时，退出码取**最严重**的一份（`5` > `4` > `3` > `2` > `1` > `0`），并在报告里逐份列出结果。

## 4. 标准输出契约

报告为人类可读文本（非 JSON）。**必须**包含以下行，供验收脚本 grep：

```text
[1/6] 校验产物 … 65 个 chunk，V1–V12 全部通过
[2/6] 连接 Milvus … http://localhost:19530  milvus v2.6.9  pymilvus x.y.z
[3/6] 版本门禁 … 集合为空，无门禁 / 一致 / 不一致（差异：…）
[4/6] 取旧数据 … doc_id=d6da41b5d356 现有 0 行（无需备份）
[5/6] 写入 … 删除 0 行，插入 65 行，校验 65 == 65 ✓
[6/6] 索引清单 … 已更新（documents: 1，total_chunks: 65）

===== 汇总 =====
  collection        med_rag_v1
  索引 / 度量        FLAT / COSINE
  维度              1024
  本次写入          65 行
  库内总行数        65
  pipeline_config_hash  <hash>
  结果              d6da41b5d356: 提交成功
```

**脱敏**：MUST NOT 打印 `MILVUS_TOKEN` 的值（宪法原则 III）。缺失时只报变量名。

## 5. 幂等契约

| 场景 | 期望 |
|---|---|
| 同一输入连跑两次 | 两次退出码 `0`；库内行数第一次后为 65，第二次后**仍为 65**；`chunk_id` 集合两次相同 |
| 库内有旧数据 + 同样的输入 | 同上（先删后插，净效果为零） |
| 库内有旧数据 + 输入变了但 `pipeline_config_hash` 不变 | 先删后插，结果反映新输入 |
| 输入变了且 `pipeline_config_hash` 变了 | 退出码 `4`，库**一字未改** |

## 6. `index_manifest.json` 的对外契约

**位置**：`med_rag\data\index_manifest.json`（与 `docs/04 §9.3` 一致；见 §2 的目录布局）。

结构见 `data-model.md §3`。消费方（后端启动校验）可以依赖：

- `pipeline_config_hash` 与 collection 内任意一行的同名字段**恒等**；
- `total_chunks` 等于 collection 内 `count(*)`；
- `documents[].chunk_count` 之和等于 `total_chunks`；
- `documents[]` 中每个 `doc_id` 在 collection 内都有 ≥ 1 行；
- 反之，collection 内的每个 `doc_id` 都在 `documents[]` 中出现。

最后两条是 **FR-022a 的核心**：manifest 与库互为镜像，不存在单方面的条目。
