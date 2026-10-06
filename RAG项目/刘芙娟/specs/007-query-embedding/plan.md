# Implementation Plan: 提问落盘与问题向量化（S8）

**Branch**: `007-query-embedding` | **Date**: 2026-09-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-query-embedding/spec.md`

---

## Summary

把用户提问变成**与知识库同一空间**的向量，并连同问题原文一起落盘。

实现上只做三件事：**复用** `backend/embed/model.py` 的 `Encoder`（不新建编码层）、在服务启动期加载权重并**校验编码指纹**、把每条提问连同向量以**一行一个 JSON 对象**追加到 `data/questions/{yyyymmdd}.jsonl`。

规模不大，但有一处**必须提前说清楚**：本特性会改变正在运行的服务 —— 启动从 < 1 s 变成 **7–11 s**，进程私有内存从数十 MB 变成 **3.26 GB**。这不是可以事后补的细节，它影响每一次开发时的重启体验。

全部抉择与实测依据见 [research.md](./research.md)；格式契约见 [contracts/store.md](./contracts/store.md) 与 [contracts/cli.md](./contracts/cli.md)。

---

## Technical Context

**Language/Version**: Python 3.12.14 —— 唯一受支持的解释器 `D:/zg6_Project/9/med_rag/rag/python.exe`（constitution 原则 I，NON-NEGOTIABLE）

**Primary Dependencies**: **本特性新增依赖为 0**。复用 `torch` / `transformers` / `numpy`（S5 已在用，实测已装）与 `backend/embed/model.py`（specs/004 的产物）。

**Storage**: 追加写 JSONL，`data/questions/{yyyymmdd}.jsonl`。无数据库。

**Testing**: **无测试框架**。禁用 pytest（`docs/superpowers/plans` Global Constraints，`specs/005` plan.md §145）。验证走「逐项对拍 + 边界构造」，产物落 `.smoke_out/`。清单见 [quickstart.md](./quickstart.md)。

**Target Platform**: Windows 本机进程，`127.0.0.1:8000`。

**Project Type**: Web 应用（同源前后端 + 离线 CLI，共用一份编码实现）

**Performance Goals**（全部为**实测值**，非估算）：

| 项 | 实测 | 预算（`docs/05` §2.4 / SC-008） |
|---|---|---|
| 服务启动（含模型加载 + 门禁） | **7.1 s**（Phase 5 实测；Phase 0 探针为 7.4 / 11.1 s，差异来自内存压力） | — （一次性） |
| 单次编码（预热后） | **0.081 – 0.092 s** | 0.3 s ✅ |
| **提问增加的耗时**（SC-008 的指标） | 中位 **0.084 s**、最差 0.092 s | ≤ 1 s ✅ |
| 端到端首字节 | 中位 **0.129 s** | — |
| 序列化单条记录 | 0.0011 s（写盘不是瓶颈） | — |
| 进程私有内存 | 3,259 MB | — （无既有预算，本次建立基线） |
| 单条记录体积 | **22,213 字节** | — （起草时估 10 KB，实测翻倍，规格已修正） |

**Constraints**:
- 单文件 ≤ 300 行（FR-023），对外入口唯一
- 全程类型注解 + snake_case（constitution 代码规范）
- 模型 MUST 本地加载（`local_files_only`），MUST NOT 触发网络下载（FR-026）
- 向量空间一致性是**硬门禁**：指纹不一致即拒绝，MUST NOT 静默继续

**Scale/Scope**: 单机单进程。预估每天数十至数百条提问 → **2.1 MB/天、约 64 MB/月**（实测每行 21–22 KB）。新增后端 6 个文件 + 2 处既有文件改动。

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 本方案的落实 | 状态 |
|---|---|---|
| **I. 环境锁定与依赖治理** | 全部命令以 `rag/python.exe` 绝对路径书写；**新增依赖 0**（复用 S5 已装的 torch/transformers/numpy）；模型权重来自本机目录，`local_files_only=True`，MUST NOT 触发网络下载 | ✅ |
| **II. 无据不答与强制溯源引用** | 本特性不产出任何面向用户的文本，不触碰回答内容。它做的是**为将来的溯源提供基础**：查询向量与文档向量同空间，是"检索到的原文确实相关"这一前提 | ✅ |
| **III. 密钥零硬编码** | 无任何密钥（模型是本地文件）。`error` 字段与日志 MUST NOT 含堆栈、路径、异常类名 | ✅ |
| **IV. 紧急症状前置响应** | 不涉及 —— 本特性不产生回答 | ✅ |
| **V. 面向群众的医疗安全边界** | 不产生面向用户的文本。**但有一处需要点名**：留存会把用户的医疗提问写入磁盘，这是本项目**第一次持久化用户输入**。与 `docs/01` §7.2「不做历史记录」的张力已由裁决 D3 处理（加注记 + 清理入口 + 不入版本库），并作为交付物 FR-028~FR-030 | ⚠️ 见下 |

**门禁结论：无违规项，无需 Complexity Tracking。**

**一处必须记录在案的张力（不构成违规）**：

constitution 原则 V 管的是"回答的边界"，本特性不产生回答，因此不在其字面范围内。但 `docs/01` §7.2 把「用户账号、登录、历史记录」列为 MVP **不做**，理由是"少收集个人信息也降低合规负担"——而本特性要做的正是持久化用户提问。

裁决 D3 的处置是**显式化 + 可退出**，不是消除：

- `docs/01` §7.2 增加注记，说明「不做历史记录」指的是面向用户的会话历史（FR-030）；
- 提供按日期清理入口（FR-028）；
- 留存目录不入版本库（FR-029）。

**这三条是本特性的交付物，不是可选项。** 若只写代码不去改那份文档，就留下了"文档说不做、代码在做"的矛盾 —— 而后来的维护者无从判断哪个算准。

---

## Project Structure

### Documentation (this feature)

```text
specs/007-query-embedding/
├── plan.md              # 本文件
├── spec.md              # 规格（已通过质量校验）
├── research.md          # Phase 0：实测基线 + R1–R7
├── data-model.md        # Phase 1：JSONL 行结构、状态机、并发前提
├── quickstart.md        # Phase 1：10 节验收清单
├── contracts/
│   ├── store.md         # 留存文件格式契约
│   └── cli.md           # CLI 契约（子命令、退出码）
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 输出（由 /speckit-tasks 生成）
```

### Source Code (repository root)

```text
backend/
├── serve.py                 # ★ 改：启动期加载模型 + 门禁 + 三行启动输出
├── embed_chunks.py          # S5 离线入口（不动）
├── embed/                   # S5 实现包（不动的依赖；本特性只 import）
│   ├── __init__.py          #   MODEL_DIR / MAX_LENGTH / EmbedError 等
│   └── model.py             #   ★ Encoder + fingerprint —— 全项目唯一编码口径
├── query_embed.py           # ★ 新增：CLI 唯一入口（embed / verify / purge / show）
├── query/                   # ★ 新增：本特性实现包
│   ├── __init__.py          #   常量：路径模板、字段名、退出码、QueryError
│   ├── gate.py              #   指纹门禁（读 index_manifest.json + 比对）
│   ├── store.py             #   JSONL 追加写 / 读取 / 原子回写 / 按日期清理
│   └── service.py           #   常驻 Encoder + capture_and_embed（服务端与 CLI 共用）
└── api/                     # specs/006 的运行时服务
    ├── capture.py           # ★ 新增：提问采集的接缝（失败不拖垮请求）
    └── ...                  #   其余不动

frontend/                    # 不动
data/
├── questions/               # ★ 新增产物目录（入 .gitignore）
└── index_manifest.json      # 门禁比对基准（只读）
```

**Structure Decision**：

沿用 `backend/` 既有的「**根级入口脚本 + 同名实现包**」形状：`chunk_clean.py`↔`chunk/`、`index_milvus.py`↔`index/`、`embed_chunks.py`↔`embed/`，本特性是 `query_embed.py`↔`query/`。

**唯一打破这个形状的地方**是本特性的 `query/` **同时被 CLI 与运行时服务引用** —— 别的实现包都只服务离线管线。这不是疏漏，是 FR-032（服务端与脚本 MUST 用同一份编码实现）的直接后果，也是 `backend/embed/model.py` 的文档字符串在两年前就预告过的用法。

### `backend/query/` 的模块切分理由

切分依据是 **"谁在什么时机可能出错"**，不是"代码属于哪一层"：

| 模块 | 只做 | 为什么单独切出来 |
|---|---|---|
| `__init__.py` | 常量与契约：路径模板、JSONL 字段名、退出码、`QueryError` | 让字段名只有一处定义 —— 写入方与读取方引用同一份常量，改名不会只改一半 |
| `gate.py` | 读清单、比对、报错 | 唯一"读别人的产物来判断自己能不能干活"的地方；可单独测（quickstart §4 只测它） |
| `store.py` | 文件读写与清理 | 唯一碰文件系统的地方；**并发前提写在这里的注释里**（见 data-model §6） |
| `service.py` | 常驻 Encoder + 组装一条记录 | 服务端与 CLI 的**共用实现**；单条与批量走同一函数，两条路径不可能漂移 |
| `api/capture.py` | 把提问交给 service，处理失败 | 让 `routes.py` 保持"只做协议转换"（specs/006 FR-015）；"失败不拖垮请求"的决策集中一处 |

---

## 实施顺序（供 tasks 阶段展开）

1. **契约常量与错误类型**（`query/__init__.py`）—— 先冻结字段名与退出码，其余都引用它；
2. **门禁**（`query/gate.py`）—— 最先做，因为它是唯一"事后无法补救"的一项，且不依赖模型能加载；
3. **存储**（`query/store.py`）—— 追加写 + 原子回写 + 清理；
4. **服务层**（`query/service.py`）—— 常驻 Encoder + `capture_and_embed`；
5. **CLI**（`query_embed.py`）—— 四个子命令；
6. **CLI 验收**（quickstart §4/§5/§7/§8 的一部分）—— **在改服务端之前**跑通 batching 与门禁，确保服务侧调试时面对的是已确认的实现；
7. **服务集成**（`api/capture.py` + `routes.py` 一行调用 + `serve.py` 启动流程）；
8. **服务验收**（quickstart §1/§2/§3/§6/§8）；
9. **文档回填**（`docs/01` §7.2 注记、`.gitignore`、`requirements.txt` 的零新增说明）。

**第 6 步不可跳过**：在门禁与批量路径未经确认前改服务端，会把"门禁写错了"和"服务集成写错了"混成同一个症状（服务起不来）。

---

## 与既有文档的差异（需同步修订）

| 文档 | 位置 | 现状 | 应改为 |
|---|---|---|---|
| `docs/01_需求分析.md` | §7.2 | 「MVP 不做…历史记录」，理由"少收集个人信息也降低合规负担" | 增加注记：该条指**面向用户的会话历史**；管线侧的提问留存另见 `specs/007`，用途限定 + 清理方式（FR-030，**本特性的交付物**） |
| `docs/05_接口设计.md` | §4.2 `retrieval_service.search` | 定义了下游检索接口 | 无改动，但需注意本特性交付的向量正是它的输入；检索接入时 `search` 可直接消费 |
| `docs/02_架构图.md` | §10 部署视图 | 已画「BGE-M3 权重加载 常驻内存」 | 与实现一致，**无需改动** —— 这是 specs/006 修订时恰好写对的一处 |
| `.gitignore` | — | 已含 `data/` 的若干子目录 | 增加 `data/questions/`（FR-029） |
| `requirements.txt` | 末尾 | specs/006 的零新增说明 | 追加 specs/007 的零新增说明（复用 S5 已登记的 torch/transformers/numpy） |

---

## 风险与已知局限

| 项 | 性质 | 处置 |
|---|---|---|
| **运行期索引重建导致指纹陈旧** | 已知局限 | 服务只在启动期校验一次（R3）。**重新入库后 MUST 重启服务**——写进 quickstart §9。正解是 `docs/05` §3.3 的 `/health` 中 `matches_index` 检查，不在本特性范围 |
| **本机内存紧张** | 环境事实 | 实测可用 1.41 GB、进程私有 3.26 GB，靠页面文件兜住。空闲 45 s 后编码未退化（实测），但**长时间空闲后是否退化未验证** —— 如实记录，不假设 |
| **单进程并发前提** | 前提，非保证 | 写入路径无 `await` + 单 worker。**启用 `--workers N` 即失效**，需改用文件锁。前提写在 `store.py` 注释里（data-model §6） |
| **每行 21 KB** | 设计代价 | 粒度是"一行一问题"，体积是换取"可 grep + 可独立判定口径"的代价。每日 100 问 ≈ 2.1 MB |

---

## Complexity Tracking

> 本次 Constitution Check 无违规项，本节留空。

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
