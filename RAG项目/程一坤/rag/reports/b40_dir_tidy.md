# 项目目录整理报告（批次 40）

- 时间：2026-09-23 14:03~14:30
- 范围：`C:\Users\92842\Desktop\rag` 全仓（排除 `frontend/node_modules`、`frontend/.next`）
- 方式：先全库扫描取证 → 分级（可删 / 不该动 / 待裁决）→ 只对「可删」动手，**一律送回收站**
- 前提：**本项目不是 git 仓库**，删除不可回滚，故未使用 `rm`，全部经 `SHFileOperationW` + `FOF_ALLOWUNDO`

## 一、结论

**目录本身是干净的**，没有结构性混乱。真正该删的只有「缓存 + 运行日志 + 构建缓存」6 类共 23 项，
其余看似可疑的地方，逐条查证后都**有明确语义，不该动**（见 §四）。

⚠️ 过程中我有一次误判并已纠正：把 `reports/latest_eval_*.{json,md}` 当成「冗余副本」删了 56 项，
查代码后发现它们是**评测工具链的约定产物**，已全部恢复（见 §三）。

## 二、已执行清理（净删 23 项，≈2.7 MB）

| 类别 | 项数 | 体积 | 依据 |
|---|---|---|---|
| `__pycache__/` + `*.pyc` | 17 目录 / 222 文件 | 2.5 MB | 字节码缓存，import 时自动重建 |
| `backend/.pytest_cache/` | 1 | — | pytest 缓存，下次运行自建 |
| `e2e_backend.log`、`e2e_review_backend.log` | 2 | 74.6 KB | `scripts/e2e/*.py` 以 `"ab"` 追加写，下次运行自动重建 |
| `frontend/restart-frontend.log` | 1 | 13.0 KB | 前端重启日志，运行产物 |
| `frontend/tsconfig.tsbuildinfo` | 1 | 93.1 KB | TS 增量编译缓存，`tsc` 自动重建 |
| `logs/`（空目录） | 1 | — | 空目录；已**恢复**（见下方说明） |

核对结果（清理后实测）：

```
剩余 __pycache__ 目录: 0
backend/.pytest_cache: 已清
e2e_backend.log / e2e_review_backend.log / frontend/restart-frontend.log / tsconfig.tsbuildinfo: 已清
backend/ 体积 3.8M → 1.3M
```

> `logs/` 我删后又**恢复**了：`scripts/deploy/run.sh:15` 定义为 `LOG_DIR="$INSTALL_DIR/logs"`，
> 且 `run.sh:46` 有 `mkdir -p "$LOG_DIR"`（技术上不需要预留），但 `install.sh:399` 会向用户提示
> "日志位置：$INSTALL_DIR/logs/"。为免任何"目录缺失"的意外，空目录已按原样建回。

## 三、自我纠错：误删 56 项 `latest_eval_*`（已全量恢复）

**我的错判**：`reports/` 里 `latest_eval_<tag>.{json,md}` 与 `eval_<时间戳>_<tag>.{json,md}` 内容 md5 完全相同
→ 我判为"冗余副本"删除。

**纠错依据（代码级）**：`evaluation/run_eval.py:191-198` 每轮**同时**写两份：

```python
json_path = args.output_dir / f"eval_{stamp}{suffix}.json"     # 历史档（带时间戳）
md_path   = args.output_dir / f"eval_{stamp}{suffix}.md"
(args.output_dir / f"latest_eval{suffix}.json").write_text(...) # 「最新一轮」指针（约定名）
(args.output_dir / f"latest_eval{suffix}.md").write_text(...)
```

同类约定还有 `calibrate_refusal.py:106,109` 写 `latest_refusal_calibration.*`、
`faithfulness.py:191,217` 写 `latest_faithfulness.*`、`selfcheck_faithfulness.py:104` 写
`latest_faithfulness_selfcheck.json`。→ **`latest_*` 是工具产物，不是冗余**。

更关键：`latest_eval_predeploy_baseline.{json,md}` 是**部署前归档件之一**（此前汇报中明确列出过），
删掉会破坏归档完整性。

**恢复结果**：56/56 全部恢复，逐个 md5 与源比对一致；`reports/` 条目数回到 **314**（与原状一致）。

## 四、查证后判定「不该动」的 4 项（避免重复讨论）

| 看起来可疑 | 查证结论 | 依据 |
|---|---|---|
| `scripts/eval/` 与 `evaluation/` 名字相近 | **刻意分工，不该合并** | `scripts/eval/README.md` 有专门对照表：`evaluation/`＝跑评测集出指标（每题独立 session）；`scripts/eval/`＝观测单条链路是否真生效（同 session 连跑多轮，摘要/改写才会触发） |
| 根 `.env` 与 `.env.development/.test/.production` 并存 | **不该改名**（名字最泛但语义明确） | `app/core/config_loader.py:48` 注释定义 `.env` 为「私有覆盖文件」；`.env.development` 头部注明"本文件不存在时回退旧单文件 `.env`" |
| `reports/` 314 个文件平铺、命名前缀不一 | **现在不该分目录** | 评测产物的输出目录由工具默认值决定（`run_eval.py:108` `--output-dir` 默认 `reports/`），细分需改代码 → 冻结期不做 |
| `data/labor_law_processed/<id>-doc-<hash>/` 目录名不可读 | **不该改** | 是导入包目录，`install.sh` 按 `--packages-root data` 扫描；且属待裁决的数据侧（旧包未替换） |

「角色不符」扫描（各目录里是否有不属于该目录角色的文件）：**7 个目录全部 0 个**。

## 五、真问题登记（待你裁决，我未改动）

### 5.1 死引用 5 处

| # | 位置 | 引用的目标 | 状态 | 类型 |
|---|---|---|---|---|
| 1 | `evaluation/faithfulness.py:128`（**默认参数**） | `reports/latest_eval.json` | ❌ 不存在 | **真 bug**：不带 `--eval-json` 跑必失败（无 tag 的那份从未生成，历轮都带 tag） |
| 2 | `evaluation/faithfulness.py:11`（docstring 示例） | 同上 | ❌ | 文档 |
| 3 | `backend/app/memory/summary_service.py:3`（注释） | `reports/batch20_session_summary_plan.md` | ❌ 已被历史清理删除 | 注释 |
| 4 | `backend/tests/test_session_summary.py:3`（注释） | 同上 | ❌ | 注释 |
| 5 | `reports/batch24b_remaining_items_acceptance.md` 等 2 个报告 | 3 个中间 eval/证据文件 | ❌ | **不改**（历史报告，改了等于篡改历史结论） |

→ 1~4 属代码/文档改动，冻结期只登记；**5 明确不改**。

### 5.2 位置可优化 1 处（低优先）

`backend/tests/demo_retrieval_pipeline.py` —— 是「验收演示脚本」（mock 数据跑检索链路），
不是测试用例：`backend/tests/` 下唯一不以 `test_` 开头的 `.py`。
**但 pytest 不收集它**（无 `test_` 前缀），且全库无任何引用（仅自身 `__main__`）→ 无害，可选处理。

## 六、整理后目录现状

```
rag/
├─ .env  .env.development  .env.test  .env.production   # 配置三件套 + 私有覆盖文件
├─ CLAUDE.md  README.md
├─ .workbuddy/                                          # 工具目录（保留）
├─ backend/   1.3M    app/ (16 子包) + tests/ (80 文件) + fixtures/ + .env.example + requirements.txt
├─ data/      4.1M    labor_law_raw/ + labor_law_processed/ + pdf_samples/ + evaluation/ + 同义词表
├─ docs/      300K    9 个文档 + archive/{plans,specs}
├─ evaluation/ 117K   18 个文件（评测主体）
├─ frontend/  538M    src/ 153K + node_modules 340M + .next 197M（构建产物，不可搬）
├─ logs/      空      部署期日志目录（run.sh 自建，已恢复）
├─ reports/   18.5M   314 个文件（报告 md / 评测 json / 证据 txt+log）
└─ scripts/   311K    deploy/ e2e/ eval/ experiments/ loadtest/ migrations/ api-collection/
```

## 七、边界

- 未改动任何业务代码、配置、数据、文档（`docs/` 按用户边界本就未动）
- 删除全部经回收站，可从回收站恢复；`%TEMP%/b40_bak/` 留有
  `b40_delete_manifest.txt`（删除清单）、`b40_delete_result.txt`（执行结果）、
  `clean_b40.py` / `restore_latest.py` / `retry_pycache.py`（本次脚本，未进 `backend/`）
- 冻结令仍生效：§五 的 1~4 项**只登记未修**
