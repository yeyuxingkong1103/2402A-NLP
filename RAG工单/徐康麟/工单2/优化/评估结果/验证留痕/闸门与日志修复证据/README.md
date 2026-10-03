# 闸门与日志修复证据（原始运行留痕归档）

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 评估结果 / 验证留痕（**证据归档目录**）
归档人：designer（t11 第 2 轮复审 F10）
归档时间：2026-10-03 17:4x
来源：工作区临时目录 `.tmp_review/`（该目录已按 captain 裁定整体删除）

---

## 1. 本目录的定位（务必先读）

- 本目录是**原始运行输出的留痕归档**，供复核者核对报告结论时**回查原始输出**；
- 本目录**不是**工单交付物清单里的条目，也不是任何一条验收的**判定依据**——
  验收判定仍以正式产物为准：`优化/评估结果/optimization_compare.md/.csv`、`accuracy_report.json`、
  `验收4_稳定性_三连跑.md`、`验收4_同类扩展样本_三连跑.md`、`优化/评估结果/验收证据矩阵.md`、
  `测试/用户/user_acceptance_checklist_v2.md`；
- 引用本目录的报告：`研发/报告/可答性闸门修复报告.md`（§5 三次运行逐题结果、§6/§8 日志与防御）。

## 2. 为什么要有这个目录（背景）

2026-10-03 首轮评审发现：`compare_optimization.py --limit 1` 曾被当作冒烟运行，
把四个正式产物覆盖成 1 题口径（F1）。清理临时件时又一度把「12 条无关问题 × 3 连跑」的
**唯一落盘原始输出**一并清掉——因为其执行脚本只 `print`、靠 shell 重定向存盘（证据落点缺陷）。

因此第 2 轮复审（t11）按 captain 裁定做两件事：

1. **把执行体转正**：`online_3runs.py` → `研发/scripts/selftest_answerability_online.py`
   （内容逻辑未改，仅补文件头工单编号与用法，脚本内**不再写入任何临时路径**）；
2. **把原始输出归档到交付树内的本目录**（正式命名、无下划线前缀），再删除 `.tmp_review/`。

## 3. 文件清单与用途

| 文件 | 产生者 | 用途 / 对应报告章节 |
| --- | --- | --- |
| `online_three_runs_逐题输出.txt` | engineer | 真实 HTTP 服务（`serve_fallback.py`，端口 8124）连跑 3 次、逐题 `is_unknown`/正文/耗时；**对应报告 §5 的“三次合计：全部 12/12”** |
| `three_runs_table_12x3.md` | engineer | 同一次三连跑的 12 题 × 3 次表格（报告 §5 表格的原始来源） |
| `selftest_production.txt` / `selftest_extractive.txt` | engineer | 闸门修复自测（生产路径 / 纯抽取式对照）全量逐题输出 |
| `selftest_production_前一批.txt` / `selftest_extractive_前一批.txt` | engineer | 上一批同源输出（`b_*`），保留供比对修复前后差异 |
| `battery_log.txt` / `battery2_log.txt` | engineer | 两批“闸门 + 日志”电池自测的运行日志 |
| `online_fault_tolerance.txt` / `online_suite_full.txt` | engineer | 在线容错与在线全量套件的原始输出 |
| `compare_full_run.txt` | engineer | `compare_optimization.py` 全量重跑的完整 stdout（含防覆盖保护提示行） |
| `log_concurrency3.txt` / `append_ab2.txt` | engineer | 日志并发追加 / 跨进程追加的实验输出 |
| `run_battery.ps1` / `run_battery2.ps1` / `run_t6_battery.ps1` | engineer | 上述电池自测的执行脚本（编排用，非交付脚本） |
| `t5_offline_full.txt` / `t5_integrity.txt` / `t5_log_newbytes.txt` | engineer | t5 收尾期的离线套件 / 交付完整性守卫 / 新增日志字节校验输出 |

> 命名说明：报告 §6/§8 曾引用 `.tmp_review/battery_log.txt` 等临时路径；
> 归档后对应文件在本目录同名（仅 `selftest_*` 两件加“前一批”后缀以区分批次）。
> 报告 §5 引用的执行脚本路径已同步为 `研发/scripts/selftest_answerability_online.py`。

## 4. 复核方法（可重跑）

```powershell
# 工作目录 = E:\gao6gongdan\工单2
# 1) 三连跑（需先启动服务；脚本不写临时路径，输出到 stdout 或自选落盘位置）
pwsh -NoProfile -File run_py.ps1 研发/app/main.py serve --port 8124
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability_online.py http://127.0.0.1:8124

# 2) 对比报告防覆盖保护（部分运行只写临时目录，见 optimization_compare.md §7）
pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py --limit 2
pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py
```
