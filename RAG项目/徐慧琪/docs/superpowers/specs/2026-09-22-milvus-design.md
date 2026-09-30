# 法律 RAG 向量化与检索索引 设计文档

> 日期：2026-09-22
> 依据：《需求文档》v1.1、《技术方案》v1.3（第 4、5、7、10 章）
> 状态：已评审通过，待转实施计划
> 范围：本文件只覆盖**子项目②向量化与索引**，不含精排、父块回填、条款号精确通路、LangChain 适配层、生成、API、前端

---

## 一、背景与本子项目的定位

入库链路（子项目①）已产出 `law_articles.jsonl`（1260 条，零缺号）与 `law_chunks.jsonl`（3388 块），但它们是**磁盘上的文本**，还不可检索。本子项目把块变成向量索引，跑通「编码 → 写库 → 混合检索」，是整个 RAG 链路第一次真正**能查**。

选它作为第二个子项目的理由：它只有入库链路一个前置依赖，且技术方案 4.3/4.4 留给本期的三处不确定（复合向量写入位置、RRF 融合位置、sparse 输出格式）**都必须靠落地实测才能定**，越早撞掉越好。

**本期新增的依据缺口**：技术方案 4.2 定义了 13 个元数据字段，`law_chunks.jsonl` 只有 5 个。缺失的 `law_id / law_version / status / effective_date / source_hash` 等按 7.1 归属 MySQL 主表——因此本期把 MySQL 一并拉进来，见 4.1。

---

## 二、运行环境（实测）

| 项 | 技术方案假设 | 本机实测 | 结论 |
|---|---|---|---|
| 容器 | Docker Compose 单机（10.1） | **Windows 无 Docker**；改为 WSL2 + `apt install docker.io` | 偏离，见第七节 |
| WSL | —— | Ubuntu 26.04.1 LTS，`D:\WSL\Ubuntu\ext4.vhdx`，用户 `xhq`，systemd 已启用 | 用户 2026-09-22 决定，规避 Docker Desktop 商业许可 |
| WSL 资源 | —— | **7.4Gi 内存 / 16 核**（WSL2 默认取宿主一半，宿主约 16G） | 内存紧张，见第九节 |
| Milvus server | v3.0.1 | **v3.0.1**（`get_server_version()` 实测） | 一致 |
| pymilvus | 3.0.2 | **3.0.2** | 一致 |
| MySQL | `mysql:8`（10.1） | 镜像 tag `mysql:8` 实装 **8.4.11** | 8.4 移除了 `mysql_native_password`，驱动需支持 `caching_sha2_password` |
| 算力 | 4090 24GB | RTX 4060 Laptop **8GB**，torch **2.13.0+cpu**（无 CUDA） | 本期编码走 CPU，见 4.3 |
| 编码模型 | bge-m3 | **`D:\Model\bge-m3`（原版）** | 见 4.3 的模型选型说明 |

**Milvus 部署形态**：`deploy/docker-compose.yml` 定义 etcd + minio + milvus + mysql 四服务，端口均绑 `127.0.0.1`（技术方案 10.1 要求"仅内网"），Windows 侧经 WSL2 的 localhost 转发访问，已实测 `127.0.0.1:19530` / `:3306` 双通。数据卷一律用 Docker 命名卷而非挂 `/mnt/d`——后者是 Windows 文件系统经 drvfs 转发，Milvus 写入频繁会被拖慢；命名卷落在 `/var/lib/docker`，即 `D:\WSL\Ubuntu\ext4.vhdx` 内，同样在 D 盘但走 ext4。

---

## 三、范围与产出

### 3.1 输入

| 文件 | 内容 |
|---|---|
| `data/parsed/law_articles.jsonl` | 1260 条法条（`number / number_cn / text / path / paragraphs`） |
| `data/parsed/law_chunks.jsonl` | 3388 块（1260 father + 1754 paragraph + 374 item） |

`law_chunks.jsonl` 的实测字段：`chunk_id`（16 位 sha1，全部同长）、`parent_id`、`chunk_type`（仅 father/paragraph/item 三种，无 leaf）、`text`（**最长 400 字**）、`article_no`、`paragraph_no`、`path`（最长 50 字）。

### 3.2 产出

1. **MySQL `fl_law` 库**：`law` / `law_version` / `article` 三表，`article` 1260 行
2. **Milvus `law_chunks` 集合**：3388 行，每行含 dense(1024) + sparse 双向量
3. **冒烟报告**：`data/parsed/smoke_report.md`，记录 N 条查询的两路命中与融合结果

### 3.3 明确不做

- **bge-reranker-v2-m3 精排**（技术方案 5.2）——留到检索链路完整时
- **父块回填去重**（FR-3.5）——本期父块已入库且带向量，回填逻辑不做
- **条款号精确通路**（FR-3.2）——本期 `article_no` 已入 Milvus 与 MySQL，通路不做
- **LangChain 适配层**（`app/adapters/lc_*.py`）——本期全走 pymilvus 直连，见 4.4
- **Redis 缓存、生成、API、前端、部署编排**

---

## 四、数据流与核心规则

```
law_articles.jsonl (1260)
        │
        ├──► MySQL article/law/law_version 表   ← 元数据权威源
        │
law_chunks.jsonl (3388)
        │
        └──► embed.py ── join MySQL(按 law_id + article_no) ──► bge-m3 双向量 ──► Milvus (3388)
                                                                                      │
                                                                        milvus.py 混合检索 + RRFRanker
```

### 4.1 MySQL 是本期的元数据权威源

技术方案 4.2 的 13 个字段中，`law_chunks.jsonl` 只有 `article_no / paragraph_no / path / text / parent_id / chunk_type` 6 个。其余按 7.1 归属 MySQL 主表，本期即建：

| 4.2 字段 | 来源 | 说明 |
|---|---|---|
| `law_id` / `law_name` | `law` 表 | 本期单行：`minfadian` / 中华人民共和国民法典 |
| `law_version` / `effective_date` / `status` | `law_version` 表 | 本期单行：`v1` / `2021-01-01` / `现行有效` |
| `article_no_cn` | `article` 表 | 来自 `law_articles.jsonl` 的 `number_cn`，4.2 要求阿拉伯/中文双存 |
| `source_hash` | `article` 表 | **新建**：`md5(条原文)`，**条级**——技术方案 4.5 要求"按 `source_hash` 重算**该条**子块向量"，故同一法条的各块共享同一个值 |
| `replaced_by` / `replaces` | `article` 表 | **建字段但留 NULL**——民法典无替代关系，这是事实不是省事 |
| `item_no` | 块级（Milvus） | 从项块正文的「（一）」前缀解析出**光杆中文数字**（存「一」，不存「（一）」），解析不出记 NULL |

**三表字段划分**（附录 C 只给了 `article` 的 DDL，`law` 与 `law_version` 按 7.1 的一行说明拆分）：

- `law`：`law_id`（主键）、`law_name`
- `law_version`：`law_id` + `law_version`（联合主键）、`effective_date`、`status`
- `article`：附录 C 的 DDL 原样，但**本期只写条级行**（1260 行），故 `paragraph_no` / `item_no` 两列在 MySQL 侧**恒为 NULL**——款/项级的编号留在 Milvus 的块级 payload 里。这不是漏填，是"主数据表存条级、块级细节存向量库"的分工

**为什么权威源放 MySQL 而不直接塞进 Milvus**：`status` / `effective_date` 要支持 SQL 范围过滤（FR-3.3、FR-3.7），条款号精确通路（FR-3.2，虽本期不做）也必须走 SQL；Milvus 侧冗余存一份是为了检索一次拿全、不回表。

### 4.2 Milvus 集合 schema

```
collection: law_chunks
  chunk_id        VARCHAR(16)          主键
  dense           FLOAT_VECTOR(1024)   HNSW / COSINE / M=16 / efConstruction=200
  sparse          SPARSE_FLOAT_VECTOR  SPARSE_INVERTED_INDEX / IP
  law_id          VARCHAR(64)          partition_key
  law_version     VARCHAR(32)
  article_no      INT64
  article_no_cn   VARCHAR(32)
  paragraph_no    INT64        nullable
  item_no         VARCHAR(16)  nullable
  path            VARCHAR(255)
  status          VARCHAR(16)          标量索引
  effective_date  VARCHAR(10)          标量索引
  parent_id       VARCHAR(16)  nullable
  chunk_type      VARCHAR(16)          标量索引
  source_hash     VARCHAR(32)
  text            VARCHAR(2000)
```

两处说明：`effective_date` 存 `YYYY-MM-DD` 字符串是 **Milvus 没有 date 类型**；`article_no` 用 INT64 而非 VARCHAR，是为了将来条款号通路能直接比大小。长度取值都按实测留了 5 倍余量（`text` 实测 400 → 2000，`path` 实测 50 → 255）。

### 4.3 编码规则

**模型选型：`D:\Model\bge-m3` 原版。** 本机另有 `bge-m3-ft-v1` 与 `bge-m3-ft-v1-r32` 两个微调版，**本期不用**，两条独立证据：

1. **域不对**：两者由 `/d/xinzg6/xm/scripts/train_embed_lora.py` 产出（`--out D:/Model/bge-m3-ft-v1`），训练数据 `eval/sets/ft_train.json` 来自 xm 项目（设备维修 RAG）自己的语料，产出日期 2026-09-01，早于本项目语料入库（2026-09-20）19 天
2. **sparse 头是坏的**：两个微调版目录**都没有 `sparse_linear.pt`**（原版有）。FlagEmbedding 找不到该文件时会**随机初始化** sparse 头，实测同一模型同一文本两次加载给出不同结果（`[17,17]` vs `[10,10]`），而原版稳定给出 `[30,30,20,20,23]`。原因是 xm 用自实现 jieba BM25、根本不用 bge-m3 的 sparse 头，故保存时未保留

用 ft-v1 会让混合检索的 sparse 路灌入**随机噪声**——不是无效，是有害。故本期用原版。

**设备：CPU。** 实测原版在 CPU 上 **2.1 块/秒**，3388 块约 **27.5 分钟**，可接受；不为它去改 `D:\Python` 的 torch 装 CUDA 轮子（8GB 显存留给后续 reranker 更划算）。

**归一化与格式**：

- dense 做 **L2 归一化**（技术方案 4.3：保证 COSINE 与内积一致）
- sparse **保留原值**，因为 Milvus 侧用 IP 度量
- FlagEmbedding 返回的 `defaultdict{token_id: weight}` 需转成普通 `{int: float}`，且**必须滤掉零权重项**——零值在稀疏倒排索引里是纯污染

**幂等**：按 `chunk_id` upsert，重跑覆盖不重复。这是选 VARCHAR 主键换来的（见第七节）。

**分批**：编码 batch 16、写入 batch 500，全程打进度——27.5 分钟没有进度输出会像卡死。

### 4.4 混合检索

```python
dense_req  = AnnSearchRequest(anns_field="dense",  param={"metric_type": "COSINE", "params": {"ef": 128}}, limit=50, expr=EXPR)
sparse_req = AnnSearchRequest(anns_field="sparse", param={"metric_type": "IP"},                   limit=50, expr=EXPR)
res = client.hybrid_search(collection, [dense_req, sparse_req], ranker=RRFRanker(k=60), limit=top_k)
```

- **过滤表达式**：`status == "现行有效" and chunk_type in ["paragraph", "item"]`
- `chunk_type` 那个过滤是"3388 块全编码"的直接后果：父块有向量，但**不能出现在召回结果里**——整条法条语义太宽，会挤掉精确的子块
- 参数取自技术方案 5.2 的"起始值"，本期不调参（正式评估集尚未建立，见第九节）

**为什么走 pymilvus 直连而不用 langchain-milvus**：技术方案 2.2「已知摩擦点 2」与 14.2 第 6 项把"复合向量写入与 RRF 融合的实现位置"列为**未决、需落地实测**，且 `langchain_milvus` 的 ranker 参数名需按落地版本核对。本期先用 pymilvus 把这条路走实，定下参数，LangChain 适配层（`lc_embeddings` / `lc_retriever`）留到接生成链路时再包——与 8.1 的模块划分一致。

---

## 五、模块划分

| 文件 | 职责 | 不做什么 |
|---|---|---|
| `deploy/mysql/schema.sql` | `law` / `law_version` / `article` 三表 DDL | **不由容器自动执行**——`mysql_data` 卷已存在，`docker-entrypoint-initdb.d` 只在数据目录为空时跑一次，现在挂进去也不会生效。改由 `load_mysql.py` 读该文件、以 `CREATE TABLE IF NOT EXISTS` 执行 |
| `backend/app/db/mysql.py` | 连接、执行、事务。本期既供 `load_mysql.py` 批量插入，也供 `embed.py` 查元数据 | 不含团队条件注入（本期无多租户） |
| `backend/app/db/milvus.py` | 集合 schema、索引、插入、混合检索封装 | **不加载任何模型** |
| `backend/app/ingest/load_mysql.py` | `law_articles.jsonl` → MySQL 三表 | 不碰 Milvus |
| `backend/app/ingest/embed.py` | `law_chunks.jsonl` + MySQL 元数据 → bge-m3 双向量 → Milvus | 不写检索逻辑 |
| `tools/smoke_retrieval.py` | 冒烟集，跑 15~20 条查询并出报告 | 不出召回率指标 |

**分层原则**：`embed.py` 只管"文本 → 双向量"，`milvus.py` 只管"schema + 读写 + 检索"。这样 `milvus.py` 的测试不必加载 2.3GB 模型，`embed.py` 的测试也不必让 Milvus 在线——两块能各自独立测。

---

## 六、错误处理

沿用入库链路的硬门槛风格（`verify_ingest.py` 定的口径）：**宁可中断，不静默跳过错数据**。

| 情形 | 处理 |
|---|---|
| 必填字段为空（`chunk_id` / `law_id` / `status` / `text`） | 中断，报出具体块 |
| 编码条数 ≠ 输入条数 | 中断（说明有块被静默丢弃） |
| sparse 转换后为空（所有词权重都是 0） | 中断——这条块的 sparse 路会失效，属于数据异常 |
| Milvus 写入失败 | 重试 3 次后抛错，**不做部分成功** |
| MySQL 连接失败 | 直接抛错，不降级为"用空元数据继续" |

**为什么不做部分成功**：入库链路已确立"错数据入库后返工代价极高"这一判断，向量库同理——3388 条里混入 N 条元数据为空的块，检索时不会报错，只会安静地给出缺字段的结果。

---

## 七、与《技术方案》的偏离

| # | 技术方案原文 | 本设计 | 理由 |
|---|---|---|---|
| 1 | 4.4 主键 `id INT64 (主键, auto_id)` | **`chunk_id VARCHAR(16)`** | `auto_id` 与 `chunk.py` 的 `_make_id()` 幂等设计**直接冲突**：重跑一次入库会生成 3388 个新 id，旧数据原样保留，集合变 6776 条且不报错；4.5 的增量 upsert 与缓存失效联动也会失效 |
| 2 | 10.1 Docker Compose | **WSL2 内 `docker.io`** | Windows 侧无 Docker；用户 2026-09-22 决定用 WSL 内 Docker（Apache-2.0），规避 Docker Desktop 商业许可 |
| 3 | 4.4 `partition_key: law_id` | 保留 | 与方案一致，且切到多法规时无需变更 schema |
| 4 | 4.3 `bge-m3` | 原版 bge-m3（非微调版） | 见 4.3 |

第 1 条需同步回改《技术方案》4.4 一行。

---

## 八、测试策略

延续"用真实语料做夹具，不造数据"的项目约定。

| 层次 | 内容 | 是否需要 Milvus |
|---|---|---|
| 纯函数单测 | sparse 转换（含零权重过滤、空输入报错）、filter 表达式构造、schema 字段定义、`article_no_cn`/`source_hash` 的 join 与计算 | 否 |
| 集成测 | 建集合 → 写 3 条真实块 → 双路检索 → 验证 RRF 融合顺序；`@pytest.mark.skipif` 无 Milvus 时跳过 | 是 |
| 冒烟 | `tools/smoke_retrieval.py` 跑 15~20 条查询，人工看命中是否相关 | 是 |

编码模型的单测**不加载真实模型**——用假的 encoder 替身验证流程，真实模型只在端到端与冒烟中出场。

---

## 九、遗留事项

1. **正式评估集未建**：需求文档 6.3 要求 50~100 条律师真实问题 + 30~50 条公众问题，且"建议一期开工前先冻结"。本期冒烟集**由 AI 照着法条内容构造**，属于"看着答案出题"，**只能证明链路通，不能作为召回率依据**。正式评估集另立任务
2. **WSL 内存 7.4Gi**：Milvus 三件套 + MySQL 常驻后余量有限，而 bge-m3 编码在宿主机侧跑，两边会抢内存。若出现容器 OOM 重启，需调 `C:\Users\35071\.wslconfig` 的上限
3. **MySQL 8.4.11 与方案 `mysql:8` 的字面差异**：8.4 移除了 `mysql_native_password`，Python 驱动需支持 `caching_sha2_password`
4. **精排 / 父块回填 / 条款号精确通路未做**：三者都在技术方案 5.2 里，本期范围外。父块已入库且带向量，回填时无需补数据
5. **8GB 显存**：本期编码走 CPU 规避了，但后续 reranker 与在线 LLM 同时上会吃紧（技术方案 10.2 已记此风险）

---

## 十、验收判据

| # | 判据 | 验证方式 |
|---|---|---|
| 1 | MySQL `article` 表 1260 行，零缺号 | `SELECT COUNT(*)` + 与 `law_articles.jsonl` 比对 |
| 2 | Milvus `law_chunks` 集合 3388 行 | `entity_count()`——**不能用 `get_collection_stats()['row_count']`**，见下 |
| 3 | 集合 schema 与 4.2 逐字段一致（含索引类型与度量） | `describe_collection` 断言 |
| 4 | dense 向量已 L2 归一化 | 抽样算模长 ≈ 1 |
| 5 | sparse 非空且无零权重项 | 抽样断言 |
| 6 | 双路检索各自能返回结果，RRF 融合结果有序 | 集成测 + 冒烟报告 |
| 7 | `chunk_type` 过滤生效：结果里不含 father | 冒烟断言 |
| 8 | 重跑入库幂等：再跑一次，集合行数仍为 3388 | 实跑第二次并比对行数 |
| 9 | 单测全绿 + `tools/check_style.py` 零告警 | 与入库链路同一质量闸门 |

> 第 8 条是本设计与技术方案 4.4 主键口径偏离的**直接验证点**——它证明幂等性确实拿到了。

> **2026-09-22 实测勘误（判据 2 的验证手段）**：`client.get_collection_stats()["row_count"]`
> 在 Milvus 3.0.1 里**不能用来数实体**，而且它不是稳定地错——是**滞后的最终一致值**：
>
> | 时刻 | `row_count` | `entity_count()` |
> |---|---|---|
> | 同一主键 upsert 两次后立刻 | 2 | 1 |
> | delete 后立刻 | 2（不降） | 0 |
> | 重跑灌库 3388 条后立刻 | 6776 | 3388 |
> | 约半小时后（分段合并完成） | **3388** | 3388 |
>
> 最后一行是本次勘误的关键：这个字段**会收敛**，所以基于它的检查会**时对时错**——
> 比稳定地错更难排查（实测者当时读到 6776，半小时后复核的人读到 3388，两人都没看错）。
>
> 这与第 8 条**直接冲突**：upsert 覆盖正是选 `chunk_id` 作主键想要的幂等行为，
> 用它为判据会时而把「重跑幂等成功」误报成「行数翻倍」。
>
> 故行数一律走 `entity_count()`，它用 `query(filter="chunk_id != ''", output_fields=["count(*)"])`，
> 反映覆盖与删除之后仍在的实体，与分段合并无关、与时点无关。
