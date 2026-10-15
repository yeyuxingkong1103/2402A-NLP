# Implementation Plan: MinerU PDF 解析（S2 解析步骤）

**Branch**: `001-mineru-pdf-parse` | **Date**: 2026-09-22 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-mineru-pdf-parse/spec.md`

## Summary

用本机已下载的 MinerU 权重，把 `data/source_data/` 中的 PDF 解析为**带页码的结构化内容产物**，落到 `data/parsed/{doc_id}/`，供后续 S3 清洗消费。

交付物是**一个单文件脚本** `backend/parse_pdf.py`（约 300 行）+ 仓库根 `requirements.txt`。脚本默认走 `pipeline` 后端（CPU 官方支持、无需额外下载、噪声类型粒度更细、可复现），保留 `--backend vlm-engine` 供切换。配置通过 `MINERU_TOOLS_CONFIG_JSON` 指向项目内的配置文件，**不修改用户全局的 `C:\Users\Lenovo\mineru.json`**。

技术决策的完整依据见 [research.md](./research.md)。

## Technical Context

**Language/Version**: Python 3.12.14（受支持运行时 `D:/zg6_Project/9/med_rag/rag/python.exe`，constitution 原则 I）

**Primary Dependencies**: `mineru>=3.2.0,<4.0`（含 PyTorch 等传递依赖，由人工按 `requirements.txt` 手动安装）。脚本本身只用 Python 标准库（`argparse`/`hashlib`/`json`/`subprocess`/`pathlib` 等）。

**Storage**: 文件系统。产物 `data/parsed/{doc_id}/`；项目内配置 `.mineru/mineru.json`

**Testing**: 本特性不引入测试框架。验证方式见 [quickstart.md](./quickstart.md)（人工执行 + 产物断言）。constitution 要求的医疗答案测试（紧急话术、免责声明）**不适用于本特性**——解析层不参与问答。

**Target Platform**: Windows 11，无 NVIDIA 显卡（仅 Intel Iris Xe 核显），CPU 推理

**Project Type**: 单文件 CLI 脚本

**Performance Goals**: 15 页原生数字版 PDF 在 CPU 下 ≤10 分钟（spec SC-001）；重复运行命中跳过时相对耗时下降 ≥90%（SC-003）

**Constraints**:
- 完全离线，禁止联网兜底下载模型（`docs/04 §11`）
- 失败必须显式，非 0 退出，禁止静默降级（constitution 代码规范）
- 单文件 ≤350 行（用户要求约 300 行）
- 不依赖库默认参数，关键参数显式写死（constitution N3）

**Scale/Scope**: 单文档 15 页起步，语料会持续增长（用户要求支持后续继续上传）。只做 S2 解析，不做清洗/分块/向量化/入库（`docs/05 §5` 关键设计 2）。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 本特性的落实 | 结论 |
|---|---|---|
| **I. 环境锁定与依赖治理** | 脚本以 `D:/zg6_Project/9/med_rag/rag/python.exe` 执行；源码内不出现裸 `python`/`python3`/`py`；所有子进程调用使用 `sys.executable` 派生路径；依赖只装进 `rag/` 并登记到根 `requirements.txt`（本特性首次创建该文件） | ✅ 通过 |
| **II. 无据不答与强制溯源引用** | 解析层是知识来源的入口。本特性**保住页码链路**（`docs/04 §5`）：产物取 `content_list.json` 而非 Markdown，因后者不携带页码；页码缺失时**报错退出**而非填 0/null | ✅ 通过 |
| **III. 密钥零硬编码** | 脚本不持有任何密钥。项目内 `.mineru/mineru.json` **只写模型路径**，不写 `bucket_info` 的 ak/sk/endpoint，不写 `llm-aided-config` 的 api_key（全局配置中的这两项均为占位符，本特性不复制它们） | ✅ 通过 |
| **IV. 紧急症状前置响应** | 解析层不参与该原则（`docs/04 §14` 已判定） | ✅ 不适用 |
| **V. 面向群众的医疗安全边界** | **有意偏离**：`docs/04 §4` 的 manifest 准入闸门（`authority_level` 限 1–2 级、`confirmed_by` 非空、sha256 比对）在本特性中不实现，改为直接由文件内容计算 `doc_id`。该偏离由用户明确指示，已在 `spec.md` 的 Assumptions 与 checklist 的"有意偏离"节记录，属**已知并接受的风险** | ⚠️ 见 Complexity Tracking |

**门禁结论**：无**未申明**的违规项。原则 V 的偏离已显式记录并给出处置建议，按 constitution 允许继续。

## Project Structure

### Documentation (this feature)

```text
specs/001-mineru-pdf-parse/
├── plan.md              # 本文件
├── spec.md              # 规格
├── research.md          # Phase 0 输出：技术决策与依据
├── data-model.md        # Phase 1 输出：产物数据结构
├── quickstart.md        # Phase 1 输出：验证指南
├── contracts/
│   └── cli.md           # Phase 1 输出：命令行契约
├── checklists/
│   └── requirements.md  # 规格质量校验
└── tasks.md             # Phase 2 输出（/speckit-tasks）
```

### Source Code (repository root)

```text
med_rag/
├── backend/
│   └── parse_pdf.py          # ★ 本特性唯一交付的代码文件
├── data/
│   ├── source_data/          # 输入：PDF 存放处（用户后续继续往这里丢）
│   │   └── 国家基层高血压防治管理指南2025版.pdf
│   └── parsed/               # 输出：解析产物，按 doc_id 隔离
│       └── {doc_id}/
│           ├── content_list.json   # 下游唯一消费对象（带 page_idx）
│           ├── middle.json         # 页级中间结构，调试用
│           ├── {name}.md           # 人工抽查用，不入管线
│           └── layout.pdf          # 版面可视化
├── .mineru/
│   └── mineru.json           # 本特性生成的项目内 MinerU 配置（覆盖全局，不修改全局）
├── requirements.txt          # 本特性首次创建
└── rag/python.exe            # 受支持运行时
```

**Structure Decision**: 采用**单项目 + 单文件脚本**结构。虽然 `docs/02 §7` 与 `docs/05 §5` 规划的最终形态是 `backend/pipeline/` 包（`-m backend.pipeline parse`），但用户明确要求单文件脚本。脚本内部按 `docs/05 §5` 的职责边界组织函数，未来平移进包结构时无需重写逻辑——这是对既有设计的有意简化，不是对它的否定。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 偏离 `docs/04 §4` 的 manifest 准入闸门（影响 constitution 原则 V 的落地） | 用户明确指示"PDF 后续还会继续上传，直接在脚本里写明存储文件在哪"，且 `data/source_data/manifest.json` 当前不存在，严格照做会导致脚本无法运行 | 保留 manifest 校验需用户先手工创建 manifest，属阻塞式前置条件；用户已选择接受该偏离。**处置建议**：在 `docs/04 §4` 补记该偏离，或在 S3 清洗前补一个独立的 `intake` 步骤把闸门加回来 |

## Phase 1 设计要点

**输入**：PDF 文件路径（单个）或目录（批量），默认目录 `data/source_data/`。

**输出**：`data/parsed/{doc_id}/`，其中 `content_list.json` 是下游唯一消费对象，每项带 `page_idx`。

**核心流程**：
1. 解析命令行参数 → 确定待处理 PDF 列表与后端
2. 对每个 PDF 计算 sha256 → 取前 12 位得 `doc_id`
3. 检查产物是否已存在且完整 → 命中则跳过（除非 `--force`）
4. 写出项目内 `.mineru/mineru.json`，设置 `MINERU_TOOLS_CONFIG_JSON` / `MINERU_MODEL_SOURCE=local` / `MINERU_DEVICE_MODE=cpu`
5. 校验权重目录存在 → 不存在则报错退出，**不联网兜底**
6. 以子进程调用 `mineru -p <pdf> -o <outdir> -b <backend>`
7. 定位产物中的 `content_list.json`，校验每项含 `page_idx` → 缺失则报错退出
8. 统计 `type` 分布，标出 `docs/04 §5` 未覆盖的类型
9. 输出人类可读的类型分布报告 + 产物路径

**退出码**（对齐 `docs/05 §5`）：`0` 成功 / `1` 参数错误 / `2` 校验失败（页码缺失、类型未知、产物不完整）/ `3` 外部依赖不可用（mineru 未安装、权重缺失）

完整的数据结构见 [data-model.md](./data-model.md)，命令行契约见 [contracts/cli.md](./contracts/cli.md)。
