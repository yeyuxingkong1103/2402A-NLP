# 响应时间基准（工单7 实测）

> 工单编号：**人工智能NLP-RAG-功能测试及评估**　|　生成时间：2026-10-07 21:33:48　|　SLA 阈值：**3000 ms**

> ⚠️ **共享 CPU 声明**：Ollama(127.0.0.1:11434) 为 CPU 单实例，与「评测官（17 题评估）」「测试官（回归）」等任务共享；本基准串行分批执行，批间休眠 5 秒，但数字仍含竞争波动，未做 CPU 独占隔离。

## 一、结论速览

| 场景 | 有效样本 | 均值 | 中位 | P50 | P90 | P95 | 最大 | **≤3000ms 达标率** | 超时题号 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cold_start` | 1 | 7339.18 | 7339.18 | 7339.18 | 7339.18 | 7339.18 | 7339.18 | **0.0**（0/1） | Q01 |
| `warm_no_cache` | 17 | 1512.48 | 1167.4 | 1167.4 | 2595.9 | 4155.03 | 4950.13 | **0.8824**（15/17） | Q10、Q13 |
| `answer_cache_hit` | 17 | 9.47 | 9.41 | 9.41 | 11.46 | 12.03 | 13.66 | **1.0**（17/17） | 无 |
| `retrieval_cache_hit` | 17 | 852.13 | 827.48 | 827.48 | 1130.72 | 1411.73 | 2126.12 | **1.0**（17/17） | 无 |
| `sqlite_fill` | 4 | 21.1 | 15.23 | 15.23 | 36.3 | 39.92 | 43.54 | **1.0**（4/4） | 无 |
| `sqlite_cache_hit` | 4 | 9.89 | 9.46 | 9.46 | 11.28 | 11.55 | 11.83 | **1.0**（4/4） | 无 |
| `probe_promptk_3` | 6 | 524.69 | 565.79 | 565.79 | 700.75 | 710.89 | 721.03 | **1.0**（6/6） | 无 |
| `probe_promptk_2` | 6 | 558.94 | 535.45 | 535.45 | 843.99 | 907.37 | 970.76 | **1.0**（6/6） | 无 |

## 二、分段耗时（均值 / P90 / 最大，ms）

| 场景 | 检索 均值 | 检索 P90 | 生成 均值 | 生成 P90 | 首字 均值 | 首字 P90 | 首字 最大 | 首字达标率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_start` | 163.65 | 163.65 | 612.9 | 612.9 | 503.01 | 503.01 | 503.01 | 1.0 |
| `warm_no_cache` | 131.58 | 172.5 | 1339.54 | 2266.74 | 703.71 | 1226.84 | 1978.54 | 1.0 |
| `answer_cache_hit` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `retrieval_cache_hit` | 0.79 | 1.41 | 835.51 | 1122.93 | 206.53 | 299.3 | 459.3 | 1.0 |
| `sqlite_fill` | 0.59 | 0.78 | 1.65 | 2.61 | 0.59 | 0.78 | 0.81 | 1.0 |
| `sqlite_cache_hit` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `probe_promptk_3` | 105.25 | 122.59 | 409.49 | 585.27 | 233.48 | 310.95 | 366.11 | 1.0 |
| `probe_promptk_2` | 96.83 | 127.78 | 448.48 | 714.22 | 270.12 | 451.01 | 623.29 | 1.0 |

## 三、逐题明细

### `cold_start`

- 说明：新进程、未预热、第 1 问（只测 1 题）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 163.65 | 612.9 | 503.01 | 7339.18 | 7340.4 | ❌ | 否 | llm_grounded | 未命中 |

### `warm_no_cache`

- 说明：warmup(with_generate=True) 后逐题首问（use_cache=False，无缓存水分）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 110.49 | 375.79 | 213.3 | 495.37 | 496.86 | ✅ | 否 | llm_grounded | 未命中 |
| Q02 | 91.09 | 943.68 | 597.04 | 1044.26 | 1045.26 | ✅ | 否 | llm_grounded | 未命中 |
| Q03 | 90.84 | 494.73 | 287.67 | 592.81 | 593.71 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 403.85 | 0.0 | 403.85 | 417.23 | 417.77 | ✅ | 是 | refused | 未命中 |
| Q05 | 138.7 | 964.67 | 883.94 | 1109.8 | 1110.42 | ✅ | 否 | llm_grounded | 未命中 |
| Q06 | 114.31 | 594.83 | 235.83 | 719.56 | 720.2 | ✅ | 否 | llm_grounded | 未命中 |
| Q07 | 91.59 | 1068.85 | 411.72 | 1167.4 | 1168.33 | ✅ | 否 | llm_grounded | 未命中 |
| Q08 | 93.06 | 1549.5 | 956.53 | 1652.97 | 1654.23 | ✅ | 否 | llm_grounded | 未命中 |
| Q09 | 118.42 | 1060.85 | 618.25 | 1187.43 | 1188.56 | ✅ | 否 | llm_grounded | 未命中 |
| Q10 | 85.81 | 4856.29 | 1978.54 | 4950.13 | 4951.17 | ❌ | 否 | rule_composer | 未命中 |
| Q11 | 97.35 | 1563.71 | 958.99 | 1670.05 | 1670.7 | ✅ | 否 | llm_grounded | 未命中 |
| Q12 | 113.21 | 1387.1 | 1024.45 | 1511.54 | 1512.16 | ✅ | 否 | llm_grounded | 未命中 |
| Q13 | 86.75 | 3321.29 | 1530.42 | 3956.26 | 3957.67 | ❌ | 否 | rule_composer | 未命中 |
| Q14 | 173.39 | 1501.52 | 706.46 | 1688.99 | 1690.41 | ✅ | 否 | llm_grounded | 未命中 |
| Q15 | 122.87 | 944.57 | 316.12 | 1079.57 | 1080.54 | ✅ | 否 | llm_grounded | 未命中 |
| Q16 | 133.28 | 800.56 | 322.1 | 944.3 | 945.59 | ✅ | 否 | llm_grounded | 未命中 |
| Q17 | 171.91 | 1344.24 | 517.8 | 1524.53 | 1525.57 | ✅ | 否 | llm_grounded | 未命中 |

### `answer_cache_hit`

- 说明：同一问题同进程第 2 次调用（use_cache=True）→ 命中答案级内存缓存

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.0 | 0.0 | 0.0 | 9.41 | 10.24 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q02 | 0.0 | 0.0 | 0.0 | 8.27 | 9.22 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q03 | 0.0 | 0.0 | 0.0 | 8.4 | 9.03 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q04 | 0.0 | 0.0 | 0.0 | 11.36 | 12.15 | ✅ | 是 | refused | 命中(memory) |
| Q05 | 0.0 | 0.0 | 0.0 | 5.73 | 6.65 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q06 | 0.0 | 0.0 | 0.0 | 8.02 | 8.83 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q07 | 0.0 | 0.0 | 0.0 | 9.61 | 10.71 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q08 | 0.0 | 0.0 | 0.0 | 10.63 | 11.58 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q09 | 0.0 | 0.0 | 0.0 | 10.4 | 11.14 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q10 | 0.0 | 0.0 | 0.0 | 8.89 | 9.74 | ✅ | 否 | rule_composer | 命中(memory) |
| Q11 | 0.0 | 0.0 | 0.0 | 9.5 | 10.12 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q12 | 0.0 | 0.0 | 0.0 | 8.14 | 9.18 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q13 | 0.0 | 0.0 | 0.0 | 13.66 | 14.41 | ✅ | 否 | rule_composer | 命中(memory) |
| Q14 | 0.0 | 0.0 | 0.0 | 9.82 | 10.61 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q15 | 0.0 | 0.0 | 0.0 | 8.59 | 9.61 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q16 | 0.0 | 0.0 | 0.0 | 8.92 | 9.67 | ✅ | 否 | llm_grounded | 命中(memory) |
| Q17 | 0.0 | 0.0 | 0.0 | 11.62 | 12.45 | ✅ | 否 | llm_grounded | 命中(memory) |

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
        "saved_ms": 2376.46,
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
        "saved_ms": 23385.08,
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
        "saved_ms": 536.95,
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
| Q01 | 0.77 | 556.16 | 284.91 | 634.01 | 635.37 | ✅ | 否 | llm_grounded | 未命中 |
| Q02 | 0.74 | 618.12 | 235.54 | 628.41 | 629.44 | ✅ | 否 | llm_grounded | 未命中 |
| Q03 | 0.72 | 499.39 | 211.13 | 510.49 | 511.94 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 0.74 | 0.0 | 0.74 | 44.44 | 45.71 | ✅ | 是 | refused | 未命中 |
| Q05 | 0.64 | 675.44 | 283.34 | 688.99 | 690.22 | ✅ | 否 | llm_grounded | 未命中 |
| Q06 | 2.2 | 679.58 | 131.02 | 693.23 | 694.27 | ✅ | 否 | llm_grounded | 未命中 |
| Q07 | 0.63 | 910.58 | 153.0 | 921.41 | 922.36 | ✅ | 否 | llm_grounded | 未命中 |
| Q08 | 0.56 | 871.37 | 183.1 | 879.53 | 880.48 | ✅ | 否 | llm_grounded | 未命中 |
| Q09 | 0.53 | 819.9 | 281.23 | 827.48 | 828.47 | ✅ | 否 | llm_grounded | 未命中 |
| Q10 | 0.95 | 1222.52 | 186.77 | 1233.13 | 1234.32 | ✅ | 否 | llm_grounded | 未命中 |
| Q11 | 1.55 | 940.81 | 206.44 | 955.72 | 956.72 | ✅ | 否 | llm_grounded | 未命中 |
| Q12 | 1.32 | 724.81 | 243.6 | 739.5 | 740.48 | ✅ | 否 | llm_grounded | 未命中 |
| Q13 | 0.43 | 2111.79 | 459.3 | 2126.12 | 2126.96 | ✅ | 否 | rule_composer | 未命中 |
| Q14 | 0.45 | 966.1 | 83.45 | 974.09 | 974.65 | ✅ | 否 | llm_grounded | 未命中 |
| Q15 | 0.51 | 828.08 | 119.44 | 838.39 | 839.1 | ✅ | 否 | llm_grounded | 未命中 |
| Q16 | 0.42 | 722.49 | 127.16 | 728.85 | 730.02 | ✅ | 否 | llm_grounded | 未命中 |
| Q17 | 0.34 | 1056.54 | 320.89 | 1062.44 | 1063.25 | ✅ | 否 | llm_grounded | 未命中 |

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
      "saved_ms": 2376.46,
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
      "saved_ms": 2376.46,
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
| Q01 | 0.81 | 1.45 | 0.81 | 43.54 | 64.01 | ✅ | 否 | cache_hit | 未命中 |
| Q02 | 0.43 | 2.49 | 0.43 | 10.39 | 27.32 | ✅ | 否 | cache_hit | 未命中 |
| Q03 | 0.7 | 2.66 | 0.7 | 11.07 | 28.85 | ✅ | 否 | cache_hit | 未命中 |
| Q04 | 0.43 | 0.0 | 0.43 | 19.39 | 35.05 | ✅ | 是 | refused | 未命中 |

### `sqlite_cache_hit`

- 说明：**另一个**引擎实例（内存缓存为空）读同一 SQLite 库 ⇒ 等价于「服务重启后重复提问」

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 0.0 | 0.0 | 0.0 | 11.83 | 22.56 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q02 | 0.0 | 0.0 | 0.0 | 8.93 | 18.32 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q03 | 0.0 | 0.0 | 0.0 | 9.99 | 18.89 | ✅ | 否 | cache_hit | 命中(sqlite) |
| Q04 | 0.0 | 0.0 | 0.0 | 8.82 | 18.79 | ✅ | 是 | refused | 命中(sqlite) |

### `probe_promptk_3`

- 说明：对照臂：RAG_RETRIEVAL__PROMPT_TOP_K=3（引擎默认）

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 99.15 | 377.55 | 237.6 | 484.09 | 485.82 | ✅ | 否 | llm_grounded | 未命中 |
| Q02 | 109.75 | 454.38 | 250.16 | 570.93 | 571.77 | ✅ | 否 | llm_grounded | 未命中 |
| Q03 | 92.2 | 454.47 | 255.79 | 560.65 | 561.6 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 114.97 | 0.0 | 114.97 | 130.95 | 131.66 | ✅ | 是 | refused | 未命中 |
| Q05 | 130.21 | 542.05 | 366.11 | 680.47 | 681.87 | ✅ | 否 | llm_grounded | 未命中 |
| Q06 | 85.25 | 628.5 | 176.27 | 721.03 | 721.61 | ✅ | 否 | llm_grounded | 未命中 |

### `probe_promptk_2`

- 说明：实验臂：RAG_RETRIEVAL__PROMPT_TOP_K=2

| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |
| Q01 | 81.88 | 369.51 | 225.22 | 485.96 | 486.58 | ✅ | 否 | llm_grounded | 未命中 |
| Q02 | 74.99 | 432.39 | 213.56 | 514.06 | 515.01 | ✅ | 否 | llm_grounded | 未命中 |
| Q03 | 80.1 | 626.22 | 278.74 | 717.21 | 717.94 | ✅ | 否 | llm_grounded | 未命中 |
| Q04 | 92.57 | 0.0 | 92.57 | 108.78 | 109.32 | ✅ | 是 | refused | 未命中 |
| Q05 | 162.99 | 802.22 | 623.29 | 970.76 | 971.45 | ✅ | 否 | llm_grounded | 未命中 |
| Q06 | 88.48 | 460.57 | 187.36 | 556.84 | 557.38 | ✅ | 否 | llm_grounded | 未命中 |

## 四、复现信息

- 复现命令：`pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py --limit 17 --top-k 10 --batch 4 --batch-sleep 5 --out 优化/评估结果/响应时间基准.json`
- 解释器：`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15）
- 工作目录：`E:\gao6gongdan\工单7`　CPU 逻辑核数：12
- 脚本启动：2026-10-07 21:31:42　结束：2026-10-07 21:33:48　总墙钟：**125.47 s**
- 预热耗时：**19930.1 ms**　明细：`{"index_chunks": 8742, "tokenizer": {"cold_ms": 0.5, "warm_ms": 0.3, "tokens": 9, "backend": "jieba"}, "entity_tagger": {"ok": true, "sample_nouns": ["武汉", "新科", "电子", "股份", "有限公司", "注册资本"], "elapsed_ms": 1.48}, "embedding": {"backend": "ollama", "model": "bge-m3:latest", "dim": 1024, "cold_ms": 25.11, "warm_ms": 23.04, "total_ms": 101.66}, "fulltext_index": {"fulltext_index": {"chunks": 8742, "fi`
- 生成模型：`qwen2.5:3b`（backend=auto）；嵌入模型：`bge-m3:latest`
- 检索 top_k：10；引擎 prompt_top_k：3；vector_k：20；bm25_k：20
- Ollama 运行前驻留：`[{"name": "qwen2.5:3b", "size": 2159374499, "size_vram": 2159374499, "expires_at": "2026-10-07T22:01:31.3248208+08:00"}, {"name": "bge-m3:latest", "size": 664000265, "size_vram": 664000265, "expires_at": "2026-10-07T21:36:30.6556464+08:00"}]`

### 被优化代码修订（SHA256 前 16 位）

| 模块 | 运行前 | 运行后 |
| --- | --- | --- |
| `retriever.py` | `45ae1b648c015d4b` | `45ae1b648c015d4b` |
| `answer_composer.py` | `6699df2bf68c033f` | `6699df2bf68c033f` |
| `generator.py` | `baa1aa93b65a974f` | `baa1aa93b65a974f` |
| `qa_engine.py` | `d5a508b0c3379bdc` | `d5a508b0c3379bdc` |
| `answerability.py` | `1f01bb8fbc79f257` | `1f01bb8fbc79f257` |
| `query_understanding.py` | `fb16916c58b8dafd` | `fb16916c58b8dafd` |
| `reranker.py` | `4679c942c4ba00f4` | `4679c942c4ba00f4` |
| `fusion.py` | `89a870a96f89cb3d` | `89a870a96f89cb3d` |
| `citation.py` | `6bb500ce28137f83` | `6bb500ce28137f83` |
| `bm25_index.py` | `bf2990e9305c470d` | `bf2990e9305c470d` |
| `cache.py` | `d72920b265f69b32` | `d72920b265f69b32` |
| `dialogue_cache.py` | `2a5a0c68e0a1c9ef` | `2a5a0c68e0a1c9ef` |

## 五、prompt_top_k 对照实验

- 子集：Q01、Q02、Q03、Q04、Q05、Q06
- 说明：{"index_chunks": 8742, "tokenizer": {"cold_ms": 0.32, "warm_ms": 0.12, "tokens": 9, "backend": "jieba"}, "entity_tagger": {"ok": true, "sample_nouns": ["武汉", "新科", "电子", "股份", "有限公司", "注册资本"], "elapsed_ms": 2.24}, "embedding": {"backend": "ollama", "model": "bge-m3:latest", "dim": 1024, "cold_ms": 29.04, "warm_ms": 20.7, "total_ms": 99.25}, "fulltext_index": {"fulltext_index": {"chunks": 8742, "fi

