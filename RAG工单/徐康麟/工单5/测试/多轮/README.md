# 测试/多轮 —— 工单5 多轮对话 Query 理解测试资产

> **工单编号**：人工智能NLP-RAG-Query 理解优化
> 本目录是**新增**的工单5 测试层，**不修改** `测试/离线`、`测试/在线`、`测试/用户`、`测试/common` 里的工单4 既有文件。
> 设计真源：`设计/接口契约_冻结.md`（`MultiTurnEngine.ask(...) -> MultiTurnAnswer`）。

---

## 1. 资产清单

| 路径 | 作用 |
| --- | --- |
| `测试/多轮/run_multiturn_eval.py` | **评测主脚本**：一次跑 4 套（多轮 / 独立会话对照 / 单轮回归 16 题 / 异常输入） |
| `测试/多轮/测试报告模板.md` | 报告模板（占位符 `{{...}}`，未跑出的一律不预填） |
| `测试/多轮/README.md` | 本文件 |
| `测试/测试数据/工单5多轮用例.jsonl` | **权威金标**：5 轮 / 2 session（MT1–MT5） |
| `测试/测试数据/工单5多轮用例_独立会话.jsonl` | 对照实验：5 轮拆成「每轮新 session」 |
| `测试/测试数据/异常输入用例.jsonl` | 12 条异常/边界输入 |
| `测试/测试数据/多轮判分.py` | **确定性判分库**（`judge_case` / `judge_abnormal_case`），不依赖网络与 LLM |
| `测试/测试数据/多轮评测结果/<run-tag>/` | 产物目录（脚本自动创建） |

> 判分**复用**工单4 的 `测试/common/golden.py`（归一化与数值等价比较取自它，单一实现来源，避免口径漂移），
> 不重造轮子；`多轮判分.py` 只在其上追加「改写/策略/焦点/跨会话泄漏」这些**多轮专有**维度。

---

## 2. 前置条件

1. **解释器**：只用工单1 的只读 venv（`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`，Python 3.11.15），
   **不要用 PATH 上的 `python`**（那是 C:\Python314，什么都没装）。
   统一走入口：`pwsh -NoProfile -File run_py.ps1 <脚本> [参数]`。
2. **核心模块**（captain 并行实现中，**本测试层不实现它们**）：
   - `研发/app/multiturn_engine.py` → `MultiTurnEngine.ask(question, *, session_id, lang, use_cache)`
   - `研发/app/core/dialogue_state.py` → `DialogueStateStore`
   - `研发/app/core/query_rewriter.py` → `QueryRewriter`
   未就绪时 `--engine local` 会给出**中文报错 + 非 0 退出码**（不会抛裸 traceback）；此时可先跑 `--suites regression,abnormal`，
   或改用 `--engine http` 打一个已启动的服务。
3. **http 模式**：服务需已启动（`部署/脚本/serve_api.py` 默认 8600；`部署/界面/web_app.py` 默认 8123）。
   脚本先探 `/health`，探不到就报中文错并退出码 3。
4. **离线能力**：`--suites multiturn,isolated` 需要检索/生成链路可用（local 模式即需要索引与 Ollama）。

---

## 3. 怎么跑

```powershell
cd E:\gao6gongdan\工单5

# ① 全量（local 直调引擎）：多轮 + 对照 + 回归 + 异常
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine local --run-tag mt-local

# ② 打服务（http）：先起服务，再指定地址
pwsh -NoProfile -File run_py.ps1 部署/脚本/serve_api.py            # 另开一个终端
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine http --base-url http://127.0.0.1:8600 --run-tag mt-http

# ③ 只跑健壮性 + 单轮回归（核心模块还没落盘时也能跑）
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine local --suites regression,abnormal --run-tag smoke

# ④ 延迟测量口径（禁用缓存）
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine local --no-cache --run-tag mt-nocache

# ⑤ 只做判分库自检（不跑网络）
pwsh -NoProfile -File run_py.ps1 测试/测试数据/多轮判分.py

# ⑥ 管道自检（核心模块没落盘时用；**合成数据，产物禁止当验收数字**）
pwsh -NoProfile -File run_py.ps1 测试/多轮/run_multiturn_eval.py --engine dry-run --run-tag dryrun-selfcheck
```

> `--engine dry-run` 是**合成数据自检运行器**：不加载模型、不检索、不生成，答案与改写全部由用例期望值回放，
> 只用来验证「判分 → 落盘 → 报告」管道（用例写错、判分口径漂移、报告渲染崩掉能在几秒内发现）。
> 防呆：`--run-tag` 必须以 `dryrun` 开头，且所有产物带 `synthetic: true` / `NOT_FOR_ACCEPTANCE: true`，
> 报告正文顶部有醒目横幅。**自检完请删掉该产物目录**，不要让合成数字留在交付物里。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--engine` | `local` | `local`=直调 `app.multiturn_engine`；`http`=POST `{base-url}/api/chat`；`dry-run`=合成数据自检 |
| `--base-url` | `http://127.0.0.1:8123` | http 模式的服务地址 |
| `--run-tag` | `mt-local` | 产物子目录名（不同口径用不同 tag，便于对照） |
| `--no-cache` | 关（即默认**启用**缓存） | 延迟测量建议加此开关；报告须标注实际取值 |
| `--out-dir` | `测试/测试数据/多轮评测结果` | 产物根目录，脚本在其下建 `<run-tag>/` |
| `--suites` | `multiturn,isolated,regression,abnormal` | 逗号分隔，可只跑其中几套 |
| `--no-warmup` | 关 | **调试口径**：跳过预热，报告须标注「冷启动未分离」 |
| `--timeout` | `180` | 单次请求超时（秒） |

---

## 4. 产物在哪、怎么看

```
测试/测试数据/多轮评测结果/<run-tag>/
├── results.jsonl      多轮逐轮：id/session/turn/改写后query/strategy/focus/answer/refused/延迟/10 个判分维度/passed
├── isolated.jsonl     独立会话对照逐轮（含 degraded_expected、counts_toward_multiturn_accuracy）
├── summary.json       准确率（逐轮 + 会话整体）/ 各维度通过率 / 策略分布 / 延迟统计 / run-meta
├── report.md          可读报告（逐轮表格；模板见 测试/多轮/测试报告模板.md）
├── regression16.json  单轮回归 16 题（judge_answer 判定 + 逐题明细）
├── abnormal.json      异常输入 12 条（逐条 + 汇总）
└── run_meta.json      运行口径：预热/冷启动/缓存/模块导入状态/起止时间
```

**看结论的顺序**：`summary.json` 的 `summary.accuracy`（逐轮）与 `summary.session_accuracy`（会话整体）
→ `summary.dimension_pass`（定位是改写没做对还是检索/生成没答对）→ `results.jsonl` 的 `reasons`（未通过原因）
→ `regression16.json`（工单4 能力是否退化）→ `abnormal.json`（健壮性）。

---

## 5. 判分口径速查（完整说明见 `多轮判分.py` 文件头）

- 准确率主口径 **`passed_turns / total_turns`**（一轮的 10 个维度全过才算通过）；
  并列 **会话整体口径**「该 session 全轮通过才算 1 个会话通过」（本工单 5 轮分属 2 个 session）。
- 数值归一化：`6,464.51 万元` ≡ `6464.51万元` ≡ `６４６４.５１ 万元`（全角/半角 + 千分位逗号 + 空白全抹）；
  **精度不放宽**（`6,464.5` ≠ `6,464.51`）。
- `must_not_contain` 带**否定护栏**：命中点前后 14 字内有否定词即豁免（与 `golden.NEGATION_CUES` 同表）。
- `no_leak`：改写后的 query 里不得出现**另一个公司**的全称（seq1 里冒出「武汉力源信息技术股份有限公司」即判错）。
- 异常输入判据 = **不崩溃 + 非空 + HTTP 不 5xx**；给出「不清楚」或友好追问都算通过，**不比对语料真值**。

---

## 6. 测量纪律（写报告时必须遵守）

1. **冷启动数字不算验收数字**：脚本先预热 1 次（单独记 `cold_start_ms`），正式跑才计时；`--no-warmup` 属调试口径，必须标注。
2. 延迟主口径 = **客户端 wall**；服务端 `latency_ms.total` 作辅助；**改写阶段单独计列**（`latency_ms.rewrite`）。
3. 统计量给 **P50 / P95 / max / min**，禁止只报平均值；逐轮判定是否 ≤3000ms 并列出超线题号。
4. 独立会话对照的 `expect_degraded=true` 轮次**不计入**多轮准确率分母（属预期降级），但必须在报告里列出。
5. **任何数字必须是实跑产物**；未跑的项写「未运行」，不得预填（`设计/接口契约_冻结.md` §3.4）。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 退出码 **2** + 「local 模式需要 `研发/app/multiturn_engine.py`，但该文件尚不存在」 | 核心模块未落盘（captain 并行开发中）。先跑 `--suites regression,abnormal`，或改用 `--engine http`。 |
| 退出码 **3** + 「服务不可用」 | http 模式没起服务，或 `--base-url` 端口不对（本脚本默认 8123，`serve_api.py` 默认 8600）。 |
| 退出码 **4** | 运行期未预期异常，stderr 里有中文说明 + traceback，请连同 traceback 反馈。 |
| 中文乱码 | 一律用 `run_py.ps1` 入口（它设了 `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1`）；直接调 `python.exe` 会乱码。 |
| 报告里全是 `未运行` | 该项对应套件没跑（`--suites` 里被排除，或模块/服务不可用）。 |
| 报告顶部出现「⛔ 合成数据警报」 | 你用了 `--engine dry-run`：那是管道自检产物，**不得当验收数字**；请删掉该目录并用 `local`/`http` 重跑。 |
| 改了用例但报告没变 | 报告是每次跑批重新生成的，不会读旧产物；请确认 `--run-tag` 是新目录。 |
