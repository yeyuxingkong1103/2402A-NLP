# Milvus 接入说明

> 日期：2026-09-16
> 状态：已接入并端到端验证通过
> 切换方式：`RAGLORA_VECTOR_STORE=qdrant|milvus|both`

---

## 一、为什么接入 Milvus

原设计用 **Qdrant 嵌入式**，当时的理由是「Milvus 需要 Docker，与『绝不重复安装』约束冲突」。

**这个理由现在不成立了**：Docker 已可用，`milvusdb/milvus:v2.6.6` 镜像（3.55GB）已在本地缓存，
且 `pymilvus 3.0.1` 与之兼容性已实测通过。同时清单明确要求 Milvus。

因此改为**双后端并存**，由配置切换，而不是替换：

| `VECTOR_STORE` | 后端 | 特点 |
|---|---|---|
| `qdrant` | 嵌入式 | 无需 Docker；但持有**独占文件锁**，后端只能单 worker |
| `milvus` | Docker standalone | 支持并发与水平扩展；需 Docker 常驻 |
| `both` | 两者都查 | 现场对比召回差异 |

**以 Qdrant 为权威源**：Milvus 数据可随时由 Qdrant 全量重建。

---

## 二、Collection Schema

对应清单要求「ID / 向量 / 原文 / 混合检索BM25 / 创建时间 / 修改时间 / 文档来源 / 摘要」：

| 字段 | Milvus 类型 | 说明 | 清单对应 |
|---|---|---|---|
| `id` | INT64, 主键, `auto_id=False` | 内容哈希（见 §四） | ID、主键、唯一性 |
| `dense` | FLOAT_VECTOR(1024) | bge-m3 稠密向量 | 向量 |
| `sparse` | SPARSE_FLOAT_VECTOR | bge-m3 learned sparse | 混合检索 BM25 |
| `text` | VARCHAR(8192) | 原文 | 向量对应的原文 |
| `source` | VARCHAR(1024) | 文档来源 | 文档来源 |
| `summary` | VARCHAR(4096) | 摘要（**当前留空**，供后续数据增强写入） | 摘要 |
| `created_at` | INT64 | 创建时间（Unix 时间戳） | 创建时间 |
| `updated_at` | INT64 | 修改时间（Unix 时间戳） | 修改时间 |
| `page` | INT64 | 页码 | *（Qdrant 侧已有，保留以对齐）* |
| `law_name` | VARCHAR(512) | 法律名 | *同上* |
| `article_no` | VARCHAR(128) | 条文号 | *同上* |

**索引**：`dense` 建 HNSW（COSINE，M=16/efConstruction=200）；`sparse` 建 SPARSE_INVERTED_INDEX（IP）。

---

## 三、实测踩到的三个坑（pymilvus 3.x 与 2.x 差异）

网上绝大多数 Milvus 示例是 2.x 写法，**照抄会踩坑**。以下三条都是本机实测：

### 坑 1：索引 API 换了入口

```python
# ✗ 2.x 写法（3.x 会失败）
from pymilvus import IndexParams          # ImportError: cannot import name
client.add_index(...)                     # AttributeError: 该方法不存在

# ✓ 3.x 写法
ip = client.prepare_index_params()
ip.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE")
client.create_index(collection_name, ip)
```

### 坑 2：多路召回要用 `hybrid_search` 而不是 `search`

```python
# ✗ 2.x 写法：把 AnnSearchRequest 列表传给 search 的 data
client.search(collection_name=..., data=[req1, req2], ranker=RRFRanker())
# -> ParamError: `search_data` value [...] is illegal

# ✓ 3.x 写法
client.hybrid_search(collection_name=..., reqs=[req1, req2], ranker=RRFRanker())
```

### 坑 3：Qdrant 稀疏向量有**两种**返回形态

| 来源 | 类型 |
|---|---|
| `query_points`（在线检索） | `dict{"indices": [...], "values": [...]}` |
| `scroll`（迁移导出） | `SparseVector` **对象**，取 `.indices` / `.values` |

只按 dict 写，迁移时抛 `TypeError: 'SparseVector' object is not subscriptable`，
而**在线检索路径看不出来** —— 因为那里用的确实是 dict。

### 另外一处（不是 3.x 的问题，是类型鸿沟）

**Qdrant 的 ID 是 64 位无符号，Milvus INT64 是有符号**，约半数 ID 会溢出：

```
DataNotMatchException: {id} field should be a int64, but got a int instead.
Detail: Value out of range: 9236543062734117978
```

处理方式见下节。

---

## 四、主键设计：为什么不用自增 ID

清单写的是「ID（主键：唯一性，**自增ID**）」，但本项目**刻意选择内容哈希**，理由：

1. **Qdrant 侧的主键本来就是内容哈希** `sha1(collection|source|article_no|text)`，
   它保证了**重复入库幂等**（同一条内容覆盖而非追加）。
2. 若改用 `auto_id=True`，重新入库同一份文档会产生**重复行** —— 这是功能退化。
3. 且迁移时无法把 Qdrant 与 Milvus 的同一条记录对应起来，无法做一致性核对。

**因此主键满足「唯一性」，但不是自增。这是有依据的取舍，不是疏漏。**

**溢出处理**：`to_milvus_id()` 按位与掉最高位，把 64 位无符号映射到 63 位有符号区间。

- 仍是**确定性函数** → 迁移可重复执行且幂等
- 但 Milvus 侧 ID 与 Qdrant 侧 ID **不相等**（差一个符号位），跨库比对需各自换算
- 碰撞风险：14955 条落在 63 位空间，生日碰撞概率约 `1e-11`；
  迁移脚本**实测校验**了无碰撞（`kb_legal 主键校验通过：14319 条唯一 ID，无碰撞`）

---

## 五、迁移

**数据直接搬运，不重新编码** —— 重编码 14955 条要跑约 4 分钟 GPU，且结果等价；
直接搬运还能保证两库逐条一致，便于核对。

```bash
# ⚠️ 必须先停后端：Qdrant 嵌入式持有独占文件锁
bash backend/shutdown.sh
cd backend
D:/anaconda3/envs/rag_env/python.exe scripts/migrate_to_milvus.py          # 迁移
D:/anaconda3/envs/rag_env/python.exe scripts/migrate_to_milvus.py --verify # 只核对
D:/anaconda3/envs/rag_env/python.exe scripts/migrate_to_milvus.py --drop   # 先删重建
bash ../backend/run.sh --bg
```

**实测结果**（2026-09-16）：

```
collection         Qdrant     Milvus  结果
kb_medical            636        636  ✓ 一致
kb_legal            14319      14319  ✓ 一致
```

迁移耗时约 9 秒。过程中同时校验了主键无碰撞。

---

## 六、验证结果

### 6.1 检索结果一致性

同一查询 `高血压的诊断标准是什么`，两库各取 top-3：

| 排名 | Qdrant | Milvus |
|---|---|---|
| 1 | 国家基层高血压防治管理指南2025版.pdf 第4页 | **同左** |
| 2 | 中国高血压防治指南2024修订版.pdf 第10页 | **同左** |
| 3 | 中国高血压防治指南2024修订版.pdf 第43页 | **同左** |

**排名完全一致**，top-5 重合 4/5。

> 分数值不同（Qdrant `0.6667/0.5/0.3333` vs Milvus `0.0318/0.0164/...`）是**正常的**：
> 两边的 RRF 常数 `k` 不同。本项目早已明确 **RRF 分是量化值、不可用于阈值过滤**
> （见 `retrieval.py` 顶部注释），相关性判断一律用**精排分**，所以这不影响结果。

### 6.3 过滤语义（两库不同，实测）

| 过滤条件 | Qdrant（精确 `MatchValue`） | Milvus（包含 `like`） |
|---|---|---|
| `law_name="民法典"` | **0 条** | **5 条** |
| `law_name="中华人民共和国民法典"` | 5 条 | 5 条 |
| `law_name="完全不存在xyz"` | **0 条** | **0 条** |

原因是库里 `law_name` 存的是**全称**（`中华人民共和国民法典`），
而调用方（`schemas.py:132` 的示例 `{"law_name": "民法典"}`）习惯传简称。

**这是刻意的行为差异，不是 bug** —— 选包含匹配是因为「传简称查不到」比「匹配略宽」
更影响可用性。若需要两库严格一致，必须统一到其中一种语义。

> 注：修复 §7.1 的过滤 bug **之前**，Qdrant 侧传任何值都返回满额结果，
> 上表第一行当时是「5 条」。修好后才是现在这个正确行为。

### 6.2 端到端评测（13 题）

| 角色 | 题数 | 来源命中 | 答案覆盖 | 引用标注 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| medical | 5 | **100%** | 100% | **100%** | 7.9s |
| legal | 5 | **100%** | 100% | **100%** | 7.5s |
| 拒答 | 3 | — | 66.7% | 66.7% | 6.0s |

**检索延迟显著改善**：混合检索 **142ms**（Milvus）vs **329ms**（Qdrant 嵌入式）。

> ⚠️ **拒答从 100% 降到 66.7%**：3 题中 1 题未守住边界。
> **样本仅 3 题，不足以判定为 Milvus 引入的回归**（单题失败即造成 33% 波动）。
> 结论需扩样本量后重测 —— 这一点如实记录，不掩盖。

---

## 七、接入过程中发现并修复的既有问题

本次接入的意外收获 —— **三个此前就存在、但一直没被发现的 bug**：

### 7.1 【严重】Qdrant 侧的检索过滤完全没生效

`retrieval._qdrant_hybrid` 把 `query_filter` 放在 `query_points` 的**顶层**，
而该查询用的是 `prefetch + FusionQuery(RRF)`。**顶层 filter 对融合不起作用**
—— 融合只合并两路排名，不重新过滤。

实测对照（传入一个绝不存在的过滤值）：

| 写法 | 结果 |
|---|---|
| 简单查询 + filter | 0 条 ✓ 过滤有效 |
| **顶层 filter + fusion（原实现）** | **5 条 ✗ 被完全忽略** |
| filter 下推到每个 `Prefetch` | 0 条 ✓ |

**影响**：`检索调试台` 传任何过滤条件都被无声忽略，结果与不过滤完全一致，
且两边都是 HTTP 200。**这个功能从写出来那天起就没工作过。**

**修复**：把 `filter` 挂到每个 `Prefetch` 上。
**回归测试**：`tests/test_retrieval_filter.py` —— 核心断言是
「传不存在的值必须返回 0 条」（仅断言「传对的值有结果」是不够的，因为过滤被忽略时传对的值同样有结果）。

### 7.2 【严重】`VECTOR_STORE=milvus` 时读写分裂

读路径按 `VECTOR_STORE` 走 Milvus，写路径（`ingest.py`）却**无条件写 Qdrant**。后果：

```
用户 POST /api/kb/ingest  ->  返回 status=ready、chunk 若干
但该文档对之后所有检索都不可见 —— 两边都不报错
```

**修复**：写路径与读路径遵守同一开关。Qdrant 仍是主写入，Milvus 为次写入；
次写入失败会**抛出明确错误**而不是静默吞掉。
删除路径（`delete_document`）同样补了 Milvus 同步删除。

**端到端验证**：入库一个测试文档 → 3 个 chunk → 立刻检索，该文档排在前两位；
删除该文档 → 两库均查不到。

### 7.3 `health.py` 仍持有 Qdrant 独占锁

`_check_qdrant()` 无条件调用 `get_client()`，会打开嵌入式 Qdrant 并拿住独占文件锁。
即便已切到 Milvus，后端仍占着 Qdrant 的锁 —— 而「换 Milvus 以解除单 worker 限制」
正是本次接入的核心收益之一。健康检查又恰恰是前端/nginx 会轮询的端点。

**修复**：`VECTOR_STORE=milvus` 时该检查标记为 `skipped`，不触碰 Qdrant。

### 7.4 `get_collection_stats` 的计数是陈旧值

Milvus 的 `get_collection_stats().row_count` **删除后只增不减**，直到后台压缩才修正。
实测删除 3 条后：`get_collection_stats` → 639，`count(*)` 聚合 → 636（正确）。

**影响**：迁移脚本的 `--verify` 原本用它比对，**删过数据后必然误报不一致**。

**修复**：新增 `milvus_store.count_rows()` 用 `count(*)` 聚合，
`--verify` 与 `collection_stats()` 全部改用它。

---

## 八、已知限制

| 限制 | 说明 | 影响 |
|---|---|---|
| **Milvus 需 Docker 常驻** | 约 2.5GB 提交量 | 本机内存紧张时需与 Dify/模拟器错开 |
| **两库字符串过滤语义不同** | Qdrant 是精确匹配（`MatchValue`），Milvus 是包含匹配（`like "%值%"`）。见 §6.3 | 传简称（`民法典`）时 Milvus 能命中、Qdrant 查不到 |
| **`summary` 字段留空** | 供后续数据增强（Plan 4）写入 | 当前为占位 |
| **跨库 ID 不相等** | 差一个符号位，需各自用 `to_milvus_id()` 换算 | 跨库比对时注意 |
| **`created_at` 是迁移时刻** | Qdrant payload 里没有时间字段，迁移时统一打上迁移时刻 | 不是文档的真实创建时间 |
| **`verify()` 只比对条数** | 不比对内容 | 条数相同但内容被截断之类的偏差查不出来 |

---

## 八、配置与运维

```bash
# 启动 Milvus（需 Docker Desktop 运行中）
bash tools/demo/up.sh milvus

# 以 Milvus 为向量库启动后端
RAGLORA_VECTOR_STORE=milvus bash backend/run.sh --bg

# 停止 Milvus 释放内存
bash tools/demo/down.sh milvus
```

**健康检查**会显示当前生效配置与各组件状态：

```bash
curl "http://127.0.0.1:8000/api/health?deep=1"
```

```json
{
  "ok": true,
  "active": {"chain": "langchain", "vector_store": "milvus", "llm_model": "qwen2.5:7b"},
  "checks": {
    "milvus": {"ok": true, "collections": ["kb_legal", "kb_medical"]}
  }
}
```

> Milvus 是**可选**后端：`VECTOR_STORE=qdrant` 时该项标记为 `skipped` 而非失败，
> Docker 未启动不会让整体健康检查变红 —— 主链路不依赖 Milvus。

---

## 九、测试

`backend/tests/test_milvus_store.py`（10 个用例）覆盖**不依赖 Milvus 服务**的纯逻辑：

- `to_milvus_id` 的无符号溢出处理、确定性、未溢出值不变、批量无碰撞
- 稀疏向量转换的两种形态（dict / 对象）
- 行转换的字段映射、缺省值归一、超长截断
- Milvus 不可达时 `health()` 优雅返回 `ok=False` 而不抛异常

> 真实读写不放进默认测试套件 —— Milvus 需 Docker，会让整套测试随 Docker
> 状态时红时绿。端到端验证由迁移脚本的 `--verify` 与在线检索完成。
