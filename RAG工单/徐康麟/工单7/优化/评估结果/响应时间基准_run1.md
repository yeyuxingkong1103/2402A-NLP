# 响应时间基准（工单7 实测）

> 工单编号：**人工智能NLP-RAG-功能测试及评估**　|　生成时间：2026-10-07 21:26:17　|　SLA 阈值：**3000 ms**

> ⚠️ **共享 CPU 声明**：Ollama(127.0.0.1:11434) 为 CPU 单实例，与「评测官（17 题评估）」「测试官（回归）」等任务共享；本基准串行分批执行，批间休眠 5 秒，但数字仍含竞争波动，未做 CPU 独占隔离。

## 一、结论速览

| 场景 | 有效样本 | 均值 | 中位 | P50 | P90 | P95 | 最大 | **≤3000ms 达标率** | 超时题号 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cold_start` | 1 | 7940.95 | 7940.95 | 7940.95 | 7940.95 | 7940.95 | 7940.95 | **0.0**（0/1） | Q01 |
| `warm_no_cache` | 17 | 1420.28 | 1419.18 | 1419.18 | 2220.01 | 2293.26 | 2303.24 | **1.0**（17/17） | 无 |
| `answer_cache_hit` | 17 | 6.06 | 6.26 | 6.26 | 7.66 | 8.53 | 10.43 | **1.0**（17/17） | 无 |
| `retrieval_cache_hit` | 17 | 1532.64 | 1390.75 | 1390.75 | 2972.15 | 3075.82 | 3119.36 | **0.8824**（15/17） | Q10、Q15 |
| `sqlite_fill` | 4 | 27.27 | 21.62 | 21.62 | 42.71 | 46.87 | 51.04 | **1.0**（4/4） | 无 |
| `sqlite_cache_hit` | 4 | 9.12 | 8.84 | 8.84 | 9.89 | 10.1 | 10.31 | **1.0**（4/4） | 无 |
| `probe_promptk_3` | 6 | 622.1 | 684.76 | 684.76 | 842.48 | 883.16 | 923.85 | **1.0**（6/6） | 无 |
| `probe_promptk_2` | 6 | 648.02 | 661.98 | 661.98 | 990.57 | 1059.35 | 1128.14 | **1.0**（6/6） | 无 |

## 二、分段耗时（均值 / P90 / 最大，ms）

| 场景 | 检索 均值 | 检索 P90 | 生成 均值 | 生成 P90 | 首字 均值 | 首字 P90 | 首字 最大 | 首字达标率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_start` | 153.24 | 153.24 | 938.93 | 938.93 | 366.74 | 366.74 | 366.74 | 1.0 |
| `warm_no_cache` | 152.9 | 177.89 | 1214.66 | 1849.65 | 434.16 | 911.66 | 1114.85 | 1.0 |
| `answer_cache_hit` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `retrieval_cache_hit` | 0.53 | 0.72 | 1522.06 | 2963.36 | 391.28 | 802.28 | 1053.35 | 1.0 |
| `sqlite_fill` | 0.83 | 1.01 | 2.42 | 4.5 | 0.83 | 1.01 | 1.1 | 1.0 |
| `sqlite_cache_hit` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `probe_promptk_3` | 98.69 | 129.22 | 514.52 | 716.96 | 230.01 | 318.12 | 365.89 | 1.0 |
| `probe_promptk_2` | 85.44 | 91.39 | 548.99 | 897.82 | 225.44 | 307.69 | 336.55 | 1.0 |

## 三、逐题明细

### `cold_start`

- 说明：新进程、未预热、第 1 问（只测 1 题）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 153.24 | 938.93 | 366.74 | 7940.95 | 7942.03 | ❌ | 否 | rule_composer | 未命中 |

### `warm_no_cache`

- 说明：warmup(with_generate=True) 后逐题首问（use_cache=False，无缓存水分）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 116.84 | 2047.13 | 880.36 | 2172.84 | 2174.87 | ✅ | 否 | rule_composer | 未命中 |
| Q02 | 121.02 | 1290.03 | 958.62 | 1419.18 | 1420.33 | ✅ | 否 | rule_composer | 未命中 |
| Q03 | 139.07 | 1283.21 | 1114.85 | 1435.22 | 1436.51 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 787.32 | 0.0 | 787.32 | 812.85 | 814.08 | ✅ | 是 | refused | 未命中 |
| Q05 | 177.6 | 963.91 | 516.71 | 1148.55 | 1149.64 | ✅ | 否 | rule_composer | 未命中 |
| Q06 | 93.97 | 654.97 | 196.63 | 756.86 | 757.49 | ✅ | 否 | llm_grounded | 未命中 |
| Q07 | 105.64 | 1717.99 | 227.41 | 1832.42 | 1833.39 | ✅ | 否 | rule_composer | 未命中 |
| Q08 | 105.64 | 1588.15 | 268.84 | 1701.75 | 1702.3 | ✅ | 否 | rule_composer | 未命中 |
| Q09 | 152.69 | 1144.7 | 487.91 | 1304.86 | 1305.73 | ✅ | 否 | rule_composer | 未命中 |
| Q10 | 107.37 | 2176.32 | 239.0 | 2290.76 | 2291.42 | ✅ | 否 | rule_composer | 未命中 |
| Q11 | 91.41 | 826.94 | 223.28 | 925.62 | 926.3 | ✅ | 否 | llm_grounded | 未命中 |
| Q12 | 87.84 | 638.15 | 242.52 | 732.53 | 733.1 | ✅ | 否 | llm_grounded | 未命中 |
| Q13 | 91.32 | 1464.34 | 137.5 | 2303.24 | 2303.99 | ✅ | 否 | rule_composer | 未命中 |
| Q14 | 80.82 | 1040.61 | 183.09 | 1131.43 | 1132.32 | ✅ | 否 | llm_grounded | 未命中 |
| Q15 | 90.88 | 1458.2 | 215.08 | 1558.25 | 1559.26 | ✅ | 否 | rule_composer | 未命中 |
| Q16 | 71.49 | 1353.68 | 193.47 | 1432.19 | 1432.86 | ✅ | 否 | rule_composer | 未命中 |
| Q17 | 178.33 | 1000.91 | 508.1 | 1186.16 | 1187.24 | ✅ | 否 | llm_grounded | 未命中 |

### `answer_cache_hit`

- 说明：同一问题同进程第 2 次调用（use_cache=True）→ 命中答案级内存缓存

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.0 | 0.0 | 0.0 | 7.4 | 8.08 | ✅ | 否 | rule_composer | 命中(memory) |
| Q02 | 0.0 | 0.0 | 0.0 | 7.37 | 7.97 | ✅ | 否 | rule_composer | 命中(memory) |
| Q03 | 0.0 | 0.0 | 0.0 | 6.26 | 6.63 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q04 | 0.0 | 0.0 | 0.0 | 4.94 | 5.46 | ✅ | 是 | refused | 命中(memory) |
| Q05 | 0.0 | 0.0 | 0.0 | 4.73 | 5.18 | ✅ | 否 | rule_composer | 命中(memory) |
| Q06 | 0.0 | 0.0 | 0.0 | 5.1 | 5.65 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q07 | 0.0 | 0.0 | 0.0 | 5.38 | 5.78 | ✅ | 否 | rule_composer | 命中(memory) |
| Q08 | 0.0 | 0.0 | 0.0 | 6.59 | 7.43 | ✅ | 否 | rule_composer | 命中(memory) |
| Q09 | 0.0 | 0.0 | 0.0 | 6.51 | 7.15 | ✅ | 否 | rule_composer | 命中(memory) |
| Q10 | 0.0 | 0.0 | 0.0 | 8.06 | 8.67 | ✅ | 否 | rule_composer | 命中(memory) |
| Q11 | 0.0 | 0.0 | 0.0 | 4.04 | 4.41 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q12 | 0.0 | 0.0 | 0.0 | 4.05 | 4.49 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q13 | 0.0 | 0.0 | 0.0 | 10.43 | 10.98 | ✅ | 否 | rule_composer | 命中(memory) |
| Q14 | 0.0 | 0.0 | 0.0 | 6.51 | 7.05 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q15 | 0.0 | 0.0 | 0.0 | 5.02 | 5.44 | ✅ | 否 | rule_composer | 命中(memory) |
| Q16 | 0.0 | 0.0 | 0.0 | 4.24 | 4.65 | ✅ | 否 | rule_composer | 命中(memory) |
| Q17 | 0.0 | 0.0 | 0.0 | 6.42 | 6.82 | ✅ | 否 | llm_grounded | 命中(memory) |

<details><summary>附加观测</summary>

```json
{
  "缓存层统计": {
    "enabled": true,
    "fingerprint": "2b38bbcd833cade9",
    "layers": {
      "normalized": {
        "name": "normalized",
        "hits": 20,
        "misses": 18,
        "puts": 18,
        "evictions": 0,
        "expired": 0,
        "saved_ms": 0.0,
        "hit_rate": 0.5263,
        "entries": 18,
        "max_entries": 512,
        "ttl_s": 3600.0
      },
      "retrieval": {
        "name": "retrieval",
        "hits": 0,
        "misses": 0,
        "puts": 18,
        "evictions": 0,
        "expired": 0,
        "saved_ms": 2729.51,
        "hit_rate": 0.0,
        "entries": 17,
        "max_entries": 512,
        "ttl_s": 3600.0
      },
      "generation": {
        "name": "generation",
        "hits": 0,
        "misses": 0,
        "puts": 17,
        "evictions": 0,
        "expired": 0,
        "saved_ms": 21588.170000000002,
        "hit_rate": 0.0,
        "entries": 16,
        "max_entries": 512,
        "ttl_s": 3600.0
      },
      "rewrite": {
        "name": "rewrite",
        "hits": 1,
        "misses": 1,
        "puts": 1,
        "evictions": 0,
        "expired": 0,
        "saved_ms": 738.96,
        "hit_rate": 0.5,
        "entries": 1,
        "max_entries": 128,
        "ttl_s": 3600.0
      }
    }
  }
}
```
</details>

### `retrieval_cache_hit`

- 说明：新引擎实例 + 生成缓存已清空，故只可能命中**检索结果缓存**；与场景 2 的 retrieve 分段对照即为检索缓存的收益

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.37 | 825.22 | 131.52 | 859.39 | 860.04 | ✅ | 否 | rule_composer | 未命中 |
| Q02 | 0.66 | 542.25 | 157.42 | 549.6 | 550.56 | ✅ | 否 | rule_composer | 未命中 |
| Q03 | 0.59 | 456.12 | 165.81 | 467.72 | 468.49 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 0.48 | 0.0 | 0.48 | 14.64 | 15.75 | ✅ | 是 | refused | 未命中 |
| Q05 | 0.37 | 649.06 | 142.2 | 657.16 | 658.6 | ✅ | 否 | rule_composer | 未命中 |
| Q06 | 0.52 | 1089.77 | 524.86 | 1098.71 | 1099.29 | ✅ | 否 | llm_grounded | 未命中 |
| Q07 | 0.4 | 2032.3 | 102.14 | 2039.9 | 2040.48 | ✅ | 否 | rule_composer | 未命中 |
| Q08 | 0.64 | 2593.69 | 438.44 | 2601.34 | 2602.46 | ✅ | 否 | rule_composer | 未命中 |
| Q09 | 0.36 | 2903.16 | 1053.35 | 2910.3 | 2911.2 | ✅ | 否 | rule_composer | 未命中 |
| Q10 | 0.45 | 3111.51 | 121.16 | 3119.36 | 3120.02 | ❌ | 否 | rule_composer | 未命中 |
| Q11 | 0.59 | 1380.45 | 683.48 | 1390.75 | 1391.45 | ✅ | 否 | llm_grounded | 未命中 |
| Q12 | 0.49 | 1068.2 | 589.95 | 1075.01 | 1075.86 | ✅ | 否 | llm_grounded | 未命中 |
| Q13 | 0.34 | 2024.15 | 570.87 | 2034.46 | 2035.72 | ✅ | 否 | rule_composer | 未命中 |
| Q14 | 0.57 | 1664.98 | 790.26 | 1674.76 | 1675.63 | ✅ | 否 | llm_grounded | 未命中 |
| Q15 | 0.82 | 3053.67 | 820.32 | 3064.93 | 3065.87 | ❌ | 否 | rule_composer | 未命中 |
| Q16 | 0.9 | 1474.77 | 148.19 | 1484.94 | 1485.55 | ✅ | 否 | rule_composer | 未命中 |
| Q17 | 0.4 | 1005.68 | 211.35 | 1011.9 | 1012.8 | ✅ | 否 | llm_grounded | 未命中 |

<details><summary>附加观测</summary>

```json
{
  "检索层缓存统计_前后": {
    "清空前": {
      "name": "retrieval",
      "hits": 0,
      "misses": 0,
      "puts": 18,
      "evictions": 0,
      "expired": 0,
      "saved_ms": 2729.51,
      "hit_rate": 0.0,
      "entries": 17,
      "max_entries": 512,
      "ttl_s": 3600.0
    },
    "清空后": {
      "name": "retrieval",
      "hits": 17,
      "misses": 0,
      "puts": 18,
      "evictions": 0,
      "expired": 0,
      "saved_ms": 2729.51,
      "hit_rate": 1.0,
      "entries": 17,
      "max_entries": 512,
      "ttl_s": 3600.0
    }
  }
}
```
</details>

### `sqlite_fill`

- 说明：新引擎实例 + 空 SQLite 库 ⇒ 真实重算并落盘（第 1 次的代价）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.8 | 1.3 | 0.8 | 51.04 | 70.16 | ✅ | 否 | cache_hit | 未命中 |
| Q02 | 0.78 | 3.43 | 0.78 | 14.81 | 33.0 | ✅ | 否 | cache_hit | 未命中 |
| Q03 | 1.1 | 4.96 | 1.1 | 19.98 | 40.81 | ✅ | 否 | cache_hit | 未命中 |
| Q04 | 0.65 | 0.0 | 0.65 | 23.27 | 39.93 | ✅ | 是 | refused | 未命中 |

### `sqlite_cache_hit`

- 说明：**另一个**引擎实例（内存缓存为空）读同一 SQLite 库 ⇒ 等价于「服务重启后重复提问」

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.0 | 0.0 | 0.0 | 10.31 | 19.75 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q02 | 0.0 | 0.0 | 0.0 | 8.78 | 17.93 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q03 | 0.0 | 0.0 | 0.0 | 8.91 | 18.71 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q04 | 0.0 | 0.0 | 0.0 | 8.48 | 18.34 | ✅ | 是 | refused | 命中(sqlite) |

### `probe_promptk_3`

- 说明：对照臂：RAG_RETRIEVAL__PROMPT_TOP_K=3（引擎默认）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 81.15 | 654.93 | 219.05 | 744.0 | 744.83 | ✅ | 否 | rule_composer | 未命中 |
| Q02 | 75.44 | 543.69 | 222.22 | 625.53 | 626.39 | ✅ | 否 | rule_composer | 未命中 |
| Q03 | 88.79 | 479.91 | 270.35 | 576.66 | 577.77 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 88.33 | 0.0 | 88.33 | 101.44 | 102.27 | ✅ | 是 | refused | 未命中 |
| Q05 | 135.19 | 779.0 | 365.89 | 923.85 | 925.05 | ✅ | 否 | rule_composer | 未命中 |
| Q06 | 123.24 | 629.57 | 214.21 | 761.1 | 762.11 | ✅ | 否 | llm_grounded | 未命中 |

### `probe_promptk_2`

- 说明：实验臂：RAG_RETRIEVAL__PROMPT_TOP_K=2

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 80.89 | 372.09 | 212.1 | 482.25 | 482.85 | ✅ | 否 | llm_grounded | 未命中 |
| Q02 | 85.32 | 448.23 | 221.3 | 541.37 | 541.96 | ✅ | 否 | llm_grounded | 未命中 |
| Q03 | 79.2 | 1040.88 | 278.82 | 1128.14 | 1129.33 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 84.43 | 0.0 | 84.43 | 100.81 | 101.63 | ✅ | 是 | refused | 未命中 |
| Q05 | 89.4 | 754.76 | 336.55 | 852.99 | 855.34 | ✅ | 否 | rule_composer | 未命中 |
| Q06 | 93.38 | 677.95 | 219.45 | 782.58 | 783.19 | ✅ | 否 | llm_grounded | 未命中 |

## 四、复现信息

- 复现命令：`pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py --limit 17 --top-k 10 --batch 4 --batch-sleep 5 --out 优化/评估结果/响应时间基准.json`
- 解释器：`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15）
- 工作目录：`E:\gao6gongdan\工单7`　CPU 逻辑核数：12
- 脚本启动：2026-10-07 21:24:00　结束：2026-10-07 21:26:17　总墙钟：**136.36 s**
- 预热耗时：**19130.93 ms**　明细：`{"index_chunks": 8742, "tokenizer": {"cold_ms": 0.31, "warm_ms": 0.18, "tokens": 9, "backend": "jieba"}, "entity_tagger": {"ok": true, "sample_nouns": ["武汉", "新科", "电子", "股份", "有限公司", "注册资本"], "elapsed_ms": 1.41}, "embedding": {"backend": "ollama", "model": "bge-m3:latest", "dim": 1024, "cold_ms": 18.63, "warm_ms": 42.23, "total_ms": 113.11}, "fulltext_index": {"fulltext_index": {"chunks": 8742, "`
- 生成模型：`qwen2.5:3b`（backend=auto）；嵌入模型：`bge-m3:latest`
- 检索 top_k：10；引擎 prompt_top_k：3；vector_k：20；bm25_k：20
- Ollama 运行前驻留：`[{"name": "qwen2.5:3b", "size": 2159374499, "size_vram": 2159374499, "expires_at": "2026-10-07T21:53:29.3256669+08:00"}, {"name": "bge-m3:latest", "size": 664000265, "size_vram": 664000265, "expires_at": "2026-10-07T21:28:59.6381025+08:00"}]`

### 被优化代码修订（SHA256 前 16 位）

| 模块 | 运行前 | 运行后 |
| --- | --- | --- |
| `retriever.py` | `45ae1b648c015d4b` | `45ae1b648c015d4b` |
| `answer_composer.py` | `6699df2bf68c033f` | `6699df2bf68c033f` |
| `generator.py` | `1be2c097ff899eee` | `baa1aa93b65a974f` |
| `qa_engine.py` | `4dc387e8c6b1c65e` | `4dc387e8c6b1c65e` |
| `answerability.py` | `1f01bb8fbc79f257` | `1f01bb8fbc79f257` |
| `query_understanding.py` | `fb16916c58b8dafd` | `fb16916c58b8dafd` |
| `reranker.py` | `4679c942c4ba00f4` | `4679c942c4ba00f4` |
| `fusion.py` | `89a870a96f89cb3d` | `89a870a96f89cb3d` |
| `citation.py` | `6bb500ce28137f83` | `6bb500ce28137f83` |
| `bm25_index.py` | `bf2990e9305c470d` | `bf2990e9305c470d` |
| `cache.py` | `d72920b265f69b32` | `d72920b265f69b32` |
| `dialogue_cache.py` | `2a5a0c68e0a1c9ef` | `2a5a0c68e0a1c9ef` |

> ⚠️ 以下模块在本次运行期间被修改，结果不可严格归因到单一修订：generator.py

## 五、prompt_top_k 对照实验

- 子集：Q01、Q02、Q03、Q04、Q05、Q06
- 说明：{"index_chunks": 8742, "tokenizer": {"cold_ms": 0.18, "warm_ms": 0.1, "tokens": 9, "backend": "jieba"}, "entity_tagger": {"ok": true, "sample_nouns": ["武汉", "新科", "电子", "股份", "有限公司", "注册资本"], "elapsed_ms": 0.94}, "embedding": {"backend": "ollama", "model": "bge-m3:latest", "dim": 1024, "cold_ms": 25.91, "warm_ms": 20.02, "total_ms": 96.45}, "fulltext_index": {"fulltext_index": {"chunks": 8742, "fi

