# Phase 1 Quickstart: 入库（S6 步骤）

**Feature**: `005-milvus-ingest` | **Date**: 2026-09-27

本文件是**验证指南**：按顺序跑，每一步都有可断言的期望值。它不是实现说明——实现细节在
`tasks.md` 与代码里。

---

## 0. 前置条件

| 项 | 要求 | 如何确认 |
|---|---|---|
| Milvus | v2.6.x standalone 在 `http://localhost:19530`，**已启动** | `docker ps --format '{{.Image}}'` 应含 `milvusdb/milvus:v2.6.9` |
| collection | 允许为空（首次）也允许已有数据 | `curl -s http://localhost:19530/v2/vectordb/collections/list -X POST -H "Content-Type: application/json" -d '{}'` |
| 上游产物 | `data/chunks/d6da41b5d356.chunks.jsonl`（65 行）、`data/embeddings/d6da41b5d356.{npy,rows.jsonl,fingerprint.json}` | `ls data/chunks data/embeddings` |
| `pymilvus` | **未安装**，需人工安装 | 见第 1 步 |

**S2–S5 的验收状态**：S3 清洗 23/23、S5 向量化 17/17 已通过（`.smoke_out/`）。本步骤不重跑它们。

---

## 1. 安装依赖（**由人执行**，宪法「Development Workflow」要求人工确认）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m pip install pymilvus
D:/zg6_Project/9/med_rag/rag/python.exe -c "import pymilvus; print(pymilvus.__version__)"
```

**期望**：打印 2.6.x 的版本号。装完后 MUST 回写仓库根 `requirements.txt`（宪法原则 I）。

---

## 2. 只读自检：不碰库就能跑的三条

```bash
cd D:/zg6_Project/9/med_rag
./rag/python.exe backend/index_milvus.py --print-env
./rag/python.exe backend/index_milvus.py --print-config
./rag/python.exe backend/index_milvus.py --dry-run
```

**期望**：

| 命令 | 期望 |
|---|---|
| `--print-env` | 打印解释器版本 3.12.14、`pymilvus` 版本、`numpy` 版本、`http://localhost:19530`、`med_rag_v1` |
| `--print-config` | 打印 `pipeline_config` 的完整输入 dict，其中 `chunk_min_chars=300`、`chunk_max_chars=700`、`min_ratio=0.6`、`clean_rule_version=clean-v1`、`chunk_rule_version=chunk-v1+bge-m3`，以及 `embed` 子 dict 的 10 个指纹字段；并打印 `pipeline_config_hash` |
| `--dry-run` | V1–V12 全部通过；**库与 manifest 上没有任何写入** |

**断言 `--print-config` 的关键一条**：`chunk_max_chars` 必须是 **700**，不是 500。
若打印出 500，说明常量没有从 `backend/chunk/core.py` import 而是从文档抄的 —— 这正是 D2 要防的漂移。

---

## 3. 首次入库

```bash
./rag/python.exe backend/index_milvus.py d6da41b5d356
```

**期望**（`contracts/cli.md §4` 的六段报告）：

- 退出码 **0**
- `[1/6]` 行显示 `65 个 chunk`，V1–V12 全部通过
- `[3/6]` 显示"集合为空，无门禁"（collection 是新建的）
- `[5/6]` 显示 `插入 65 行，校验 65 == 65`
- 汇总里 `库内总行数 65`

---

## 4. 核对写入结果 —— SC-001 / SC-007

```bash
# 4.1 行数
./rag/python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient('http://localhost:19530')
c.load_collection('med_rag_v1')
print('rows =', c.query('med_rag_v1', filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)'])
"
# 期望 rows = 65

# 4.2 索引配置
./rag/python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient('http://localhost:19530')
print(c.list_indexes('med_rag_v1'))
print(c.describe_index('med_rag_v1', c.list_indexes('med_rag_v1')[0]))
"
# 期望 index_type=FLAT, metric_type=COSINE

# 4.3 抽 20 条逐字比对 text / page_start / section（对拍脚本见 tasks.md 的实现任务）
./rag/python.exe .smoke_out/s6_verify.py
# 期望：20/20 命中，0 处不符
```

---

## 5. 幂等 —— SC-002

```bash
./rag/python.exe backend/index_milvus.py d6da41b5d356
```

**期望**：退出码 `0`；`库内总行数` **仍是 65**（不是 130）；`[4/6]` 行显示"现有 65 行（已备份）"。

再核对一次 `chunk_id` 集合与第 4 步相同。

---

## 6. 版本门禁 —— SC-003

构造一处不一致：把库内某行的 `pipeline_config_hash` 改成一个不同的值是不现实的（它逐行一致），
因此**用反向方式验证**：临时把 `backend/chunk/core.py` 的 `HIGH` 从 `700` 改成 `701`（**改完必须改回**），
再跑一次入库。

```bash
# 改 HIGH=701 后
./rag/python.exe backend/index_milvus.py d6da41b5d356
# 期望：退出码 4；错误信息里点名 'chunk_max_chars'，给出 700 与 701 两个值
# 并且库内行数仍是 65、内容一字未改

# 还原 HIGH=700
```

两种更轻的替代构造方式（不改源码）：

```bash
# 方式二：伪造一个不匹配的 manifest，观察退出码 4
# 方式三：--print-config 得到本次 hash，与库内行里的 hash 手工比对
./rag/python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient('http://localhost:19530')
print(c.query('med_rag_v1', filter='', output_fields=['pipeline_config_hash'], limit=1, consistency_level='Strong'))
"
```

---

## 7. 回滚 —— SC-004

构造一次"插入中途失败"。最省事的办法是**注入一条超长 `text`**：

```bash
# 备份原产物
cp data/chunks/d6da41b5d356.chunks.jsonl data/chunks/d6da41b5d356.chunks.jsonl.bak
# 把第 0 行的 text 撑到 20000 字节（超过 max_length=16384）
./rag/python.exe -c "
import json
p='data/chunks/d6da41b5d356.chunks.jsonl'
rows=[json.loads(l) for l in open(p,encoding='utf-8')]
rows[0]['text']='啊'*7000
open(p,'w',encoding='utf-8').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
"
./rag/python.exe backend/index_milvus.py d6da41b5d356
# 期望：退出码 2，错误信息点出 chunk_id=d6da41b5d356:0000、字段 text、实际 21000 字节、上限 16384
#       并且 库内行数 仍是 65（因为校验在写之前，库压根没被碰）
# 还原
mv data/chunks/d6da41b5d356.chunks.jsonl.bak data/chunks/d6da41b5d356.chunks.jsonl
```

**真正的回滚路径**（插入后校验失败）难以在真实服务上稳定复现，因此由**边界构造 + 代码走查**
覆盖：实现 MUST 提供一段可注入的"伪造 insert_count"钩子（仅在环境变量 `MEDRAG_TEST_FAULT=insert_count`
下生效），用于断言"回滚被触发、库回到 65 行、退出码 2"。该钩子 MUST NOT 影响正常路径。

---

## 8. 清单身份证 —— SC-009

```bash
cat data/index_manifest.json
```

**期望**逐项与库/产物对上：

| 字段 | 期望 |
|---|---|
| `total_chunks` | 65 |
| `dim` | 1024 |
| `metric_type` / `index_type` | `COSINE` / `FLAT` |
| `documents` | **1 条**（重复跑过多次也仍是 1 条，不是 2 条） |
| `documents[0].chunk_count` | 65 |
| `documents[0].page_count` | 15 |
| `documents[0].doc_id` | `d6da41b5d356` |
| `documents[0].source_hash` | `39ced98c…79bb8`（**MinerU `_origin.pdf` 的哈希**，不是源 PDF 的 `d6da41b5d356…`） |
| `pipeline_config_hash` | 与第 6 步从库内读出的值**一致** |
| `pipeline_config.chunk_max_chars` | **700** |

---

## 9. 多文档与"留空 = 全部" —— D3

当前只有一份文档，因此断言退化为：

```bash
./rag/python.exe backend/index_milvus.py     # 不带位置参数
```

**期望**：行为与显式传 `d6da41b5d356` 完全一致；报告的 `[1/6]` 行列出扫描到的文档数为 1。

多文档的真正验证留到第 2 份语料准入后进行（`data/source_data` 目前只有 1 份）。

---

## 10. 清理（可选，验证完毕后）

```bash
./rag/python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient('http://localhost:19530')
c.drop_collection('med_rag_v1')
"
rm -f data/index_manifest.json
rm -rf data/index_backup
```

**注意**：`drop_collection` 是破坏性的。只在确认要重来时执行。
