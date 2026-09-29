# 批次 42：两份同名 `render_markdown` 加差异注释

**日期**：2026-09-23　**性质**：纯注释改动，零行为变化
**范围**：仅 `evaluation/refusal_report.py`、`evaluation/render_report.py` 各加 1 行注释
**边界遵守**：未改 `docs/`；未动业务代码；改前备份含 md5；源码用 `read_bytes/write_bytes`；临时脚本在 `%TEMP%/b42_bak/`

---

## 一、为什么需要这行注释

`evaluation/` 下有两个**同名但不同用途**的 `render_markdown(payload)`：

| 文件 | 渲染对象 | payload 顶层键 | 输出文件名 | 唯一调用方 |
|---|---|---|---|---|
| `refusal_report.py` | 拒答阈值校准报告 | `run_at` / `eval_set` / `rows` / `scans` / `chosen_threshold` / `comparison` | `refusal_calibration_<stamp>.md`、`latest_refusal_calibration.md` | `calibrate_refusal.py`（L39 导入，L105/L110 调用） |
| `render_report.py` | 评测主报告 | `run_at` / `eval_set` / `run_config` / `summary` / `worst_samples` / `details` | `eval_<stamp>_<tag>.md`、`latest_eval<tag>.md` | `run_eval.py`（L74 导入，L194/L198 调用） |

二者**不等价、不可合并**（批次 41 已出结论）：入参结构、章节骨架、调用方全不同，只是函数名撞车。
单看函数名无从判断该调哪个 → 各加一行注释，把「差异 + 调用方」写在定义处。

## 二、改动内容

```diff
--- a/evaluation/refusal_report.py
+++ b/evaluation/refusal_report.py
@@ -15,6 +15,7 @@
 from typing import Any
 
 
+# 与 render_report.render_markdown() 同名不同用途：本函数只渲染拒答校准报告，调用方仅 calibrate_refusal.py
 def render_markdown(payload: dict[str, Any]) -> str:
```

```diff
--- a/evaluation/render_report.py
+++ b/evaluation/render_report.py
@@ -24,6 +24,7 @@
 from run_config import format_run_config_lines  # noqa: E402
 
 
+# 与 refusal_report.render_markdown() 同名不同用途：本函数渲染评测主报告（summary/worst_samples），调用方仅 run_eval.py
 def render_markdown(payload: dict[str, Any]) -> str:
```

| 项 | `refusal_report.py` | `render_report.py` |
|---|---|---|
| 锚点命中数 | 1（期望 1） | 1（期望 1） |
| 新增行行宽 | 81 字符 | 94 字符（该文件既有最长行 159，习惯内） |
| 改前 md5 | `c499369dd6b86ae4e44f9aae70c226a5` | `797df3758112cfe39a2b7f93d27dfc88` |
| 改后 md5 | `14c71c73f4df…` | `696118ad0eb4…` |
| 行尾/编码 | 纯 LF，无 BOM（改前 117 行 → 118 行） | 纯 LF，无 BOM（改前 91 行 → 92 行） |
| 备份位置 | `%TEMP%/b42_bak/refusal_report.py` | `%TEMP%/b42_bak/render_report.py` |

## 三、证明链（三层，逐层加强）

### 1. 逐行 diff：各只加 1 行

`diff -u 备份 现状` 输出均为**单个 `+` 行**，无任何其他差异（见 §二）。

### 2. 逐函数 AST 对拍：函数体逐字相同

| 文件 | 函数数（前→后） | 新增 | 删除 | **函数体变化** | 模块级语句差异 | 去注释后逐行 diff |
|---|---|---|---|---|---|---|
| `refusal_report.py` | 1 → 1 | 无 | 无 | **无** | 无 | **0 行** |
| `render_report.py` | 1 → 1 | 无 | 无 | **无** | 无 | **0 行** |

并确认新注释确实落在目标函数**正上方**（`refusal_report.py` 第 19 行定义、注释在第 18 行；`render_report.py` 第 28 行定义、注释在第 27 行）。

### 3. 运行期等价：同 payload → 输出逐字节相同（最强证据）

用**磁盘上真实归档的 payload** 分别喂给改前（从备份加载）与改后函数：

| 用例 | payload 来源 | 改前输出 | 改后输出 | 逐字节相同 | 与磁盘归档 `.md` 比对 |
|---|---|---|---|---|---|
| `refusal_report.render_markdown` | `latest_refusal_calibration.json` | 1871 字符 / md5 `c09e36e0e5dc3a45a27f133164c7f23c` | 同上，md5 一致 | **YES** | **逐字节一致** |
| `render_report.render_markdown` | `latest_eval_predeploy_baseline.json` | 5538 字符 / md5 `39ed1e92180bf283c5756c553351d1fa` | 同上，md5 一致 | **YES** | **逐字节一致** |

> 第二项尤其有力：它证明**评测主报告的渲染结果与归档期完全一致**，即这份改动的产物和 `predeploy_baseline` 归档件能逐字节复现。

### 4. 全量测试：不减

```
744 passed, 3 warnings in 14.83s
```

与批次 41 基线 **744 passed** 相同，不减。前端未触碰（本轮仅 `.py` 注释），故 `npm run build` 不受影响、无需重跑。

### 5. 基线锚点未动

| 锚点 | 值 | 判定 |
|---|---|---|
| `backend/app/chat/prompt_builder.py` md5 | `f43ee63672ff2b7e136a968441bac640` | **与基线归档值一致** |
| 本轮被改源码文件 | 仅 `evaluation/refusal_report.py`、`evaluation/render_report.py` | 检索/生成/护栏路径**零触碰** |

→ **已归档的评测基线仍然有效**，无需重跑。

## 四、收尾

- 跑测试产生的 `__pycache__`×17、`backend/.pytest_cache/` 已全部送回收站（残留 0）。
- 临时脚本（`b42_edit.py` / `b42_proof.py` / `b42_runtime_equiv.py` / `b42_clean_cache.py`）均在 `%TEMP%/b42_bak/`，未进 `backend/`。

## 五、冻结期状态

代码冻结继续生效。待办清单（部署验收后处理）：

| # | 项 | 出处 |
|---|---|---|
| 1 | `MAX_PAGES` 页数拦截实现（含 monkeypatch 断言零 HTTP 请求的验收口径） | 任务 **#212** |
| 2 | `confusable` 弱项改善（Recall@5 0.6667，最弱题型） | 批次 39 基线分类型表 |
| 3 | 结果缓存与余弦阈值开关 | 用户裁决 |
| 4 | 低质量文档清理 | 用户裁决 |

部署完成后的第一件事：**重跑全量 100 题，与 `eval_20260922_212626_predeploy_baseline` 逐项对比**（先 diff `run_config`，模型/窗口/阈值/prompt md5 任一变了对比即不成立）。
