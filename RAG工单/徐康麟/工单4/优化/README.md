# 优化（T7）：缓存 / 批处理 / 靶点修复与前后对比

> 工单编号：**人工智能NLP-RAG-图像内容解析及检索优化**
> 交付报告：`优化/评估结果/优化报告.md`（逐项对比表 + 靶点 A/B/C 修前修后证据 + 未运行项登记）

## 目录

| 路径 | 内容 |
| --- | --- |
| `优化/脚本/baseline_compare.py` | **verify 命令**：对比 `优化/基线/` 与 `优化/评估结果/` 的同一协议评测结果，落盘对比 JSON/Markdown，未达标非零退出 |
| `优化/脚本/bench_cache.py` | 多级缓存基准：冷跑 vs 答案级缓存 vs 仅 T7 检索/生成缓存（三档分列） |
| `优化/脚本/bench_batch_embed.py` | 批处理基准：逐条 vs 批量嵌入（往返次数、加速比、逐行余弦一致性） |
| `优化/脚本/purge_cache.py` | 清理改动前遗留的答案级缓存（防止旧算法结果被复用；只动问答侧库） |
| `优化/基线/` | 优化前评测产物（T6 同一份 `run_16q_eval.py` 复现）+ 服务日志 |
| `优化/评估结果/` | 优化后 run1/run2（冷进程全量）、缓存命中档、`T6_评测基线.json`、三份基准 JSON、对比表、本报告 |

## 一键复现

```powershell
# 1) 优化前后对比（verify）
pwsh -NoProfile -File run_py.ps1 优化/脚本/baseline_compare.py --before 优化/基线/ --after 优化/评估结果/

# 2) 优化后评测（每次先重启服务，保证是冷进程 full-path；8123 仅本工单使用）
pwsh -NoProfile -File run_py.ps1 测试/在线/run_16q_eval.py --out-dir 优化/评估结果/run1_full --run-tag t7-after-run1 --suites main,neg,bilingual

# 3) 缓存与批处理基准
pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_cache.py
pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_batch_embed.py --n 48 --batch-size 16

# 4) 清理旧答案缓存（改了检索/生成算法后建议执行一次）
pwsh -NoProfile -File run_py.ps1 优化/脚本/purge_cache.py
```

## 结论速览（详见报告）

| 指标 | 优化前（T6 实测） | 优化后（run1 / run2） |
| --- | --- | --- |
| 准确率 | 16/16 | **16/16（未退化）** |
| 权威层命中（总 / BM25 / 向量） | 14 / 14 / 11 | 14 / 14 / **12** |
| 拒答负例 | 3/4（N1 未通过） | **4/4** |
| 英文路径 | 0/3 | **3/3（连续两次一致）** |
| 引用可回溯 | 16/16 | 16/16 |
| 延迟 wall P50 / P95 / max | 595.8 / 929.8 / 1062.8 ms | 810.1 / 1519.4 / 2017.8 ms（run2 710.9 / 1472.5 / 1971.1） |
| 超 3000 ms 题数 | 0 | **0** |
| 缓存命中档（重复提问） | 未运行 | 答案级 P50 2.2 ms（358×）/ 仅 T7 两层 P50 6.6 ms（119×） |
| 批处理嵌入（48 条） | 2593 ms（逐条） | **1009 ms（批量 16）· 2.57× · 余弦 1.000000** |

> **英文通过数的字段口径（勿误读）**：「英文路径 0/3 → 1/3 → 3/3」取自 `bilingual.jsonl` 的
> **`en_criterion_passed`**（语言无关的英文原子口径）。同文件里的 **`correct` 是中文关键字口径**
> （`golden.judge_answer`），对英文答案**天然为 False**，**不代表英文未通过**；两者并列保留、互不覆盖。
> 逐轮逐题取值见 `优化/评估结果/优化报告.md` §3 靶点 B 的来源字段表。
