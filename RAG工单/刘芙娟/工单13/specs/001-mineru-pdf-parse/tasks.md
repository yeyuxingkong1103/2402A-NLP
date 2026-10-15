# Tasks: MinerU PDF 解析（S2 解析步骤）

**Feature**: `specs/001-mineru-pdf-parse` | **Date**: 2026-09-22

**Input**: [plan.md](./plan.md)、[spec.md](./spec.md)、[data-model.md](./data-model.md)、[contracts/cli.md](./contracts/cli.md)

**交付物**：`backend/parse_pdf.py`（约 300 行，单文件）、`requirements.txt`（仓库根，首次创建）

**说明**：本特性为单文件脚本，任务按**函数粒度**拆分，不按文件拆分。全部任务在同一文件内完成。

---

## Phase 1：基础与常量

- [ ] **T-01** 文件头文档块：写清用途、输入输出、存储位置、执行方式、已锁定的技术决策、以及**对 `docs/04 §4` 的有意偏离说明**
- [ ] **T-02** 常量与路径推断：仓库根（可由 `MED_RAG_REPO_ROOT` 覆盖）、`data/source_data/`、`data/parsed/`、`.mineru/mineru.json`、两份模型权重的默认绝对路径
- [ ] **T-03** 退出码常量：`EXIT_OK=0` / `EXIT_ARG=1` / `EXIT_VALIDATION=2` / `EXIT_DEPENDENCY=3`（对齐 `docs/05 §5`）

## Phase 2：核心函数（纯逻辑，可独立验证）

- [ ] **T-04** `compute_doc_id(pdf_path) -> str`：流式读文件算 sha256，取前 12 位（`docs/04 §10.1`）
- [ ] **T-05** `collect_pdfs(input_path) -> list[Path]`：单文件或目录；目录时取 `*.pdf`；按 sha256 去重；为空则报错
- [ ] **T-06** `write_mineru_config(config_path, pipeline_dir, vlm_dir) -> None`：写 `models-dir` + `"model-source": "local"`；**不写** `bucket_info` / `llm-aided-config`（constitution 原则 III）
- [ ] **T-07** `build_mineru_env(config_path) -> dict`：构造子进程环境变量（在 `os.environ` 副本上叠加 `MINERU_TOOLS_CONFIG_JSON` / `MINERU_MODEL_SOURCE=local` / `MINERU_DEVICE_MODE=cpu`），**不修改当前进程环境**
- [ ] **T-08** `locate_content_list(parsed_dir) -> Path`：容错定位 `content_list.json`（MinerU 各版本层级不同）；找不到即抛错，**不得静默跳过**

## Phase 3：解析与校验

- [ ] **T-09** `resolve_mineru_exe() -> Path`：由 `sys.executable` 派生 `Scripts/mineru.exe`，回退 `which("mineru")`；找不到即退出码 `3`
- [ ] **T-10** `check_weights(backend, pipeline_dir, vlm_dir) -> None`：校验所选后端的权重目录存在；不存在即退出码 `1` 并提示**拒绝联网下载**（`docs/04 §11`）
- [ ] **T-11** `run_mineru(exe, pdf, out_dir, backend, env) -> None`：子进程调用 `mineru -p <pdf> -o <out_dir> -b <backend>`；非 0 退出即映射为退出码 `3`
- [ ] **T-12** `is_artifact_complete(doc_dir) -> bool`：判定产物是否可跳过——`content_list.json` 存在且非空、JSON 可解析、块数 > 0
- [ ] **T-13** `validate_page_idx(items) -> None`：**每项必须有 `page_idx`**；缺失即退出码 `2`，错误信息指出具体块索引（`docs/04 §11` 禁止填 0/null）
- [ ] **T-14** `build_type_distribution(items, backend, version) -> dict`：统计 `type` 分布、`page_idx` 范围、未知类型列表

## Phase 4：命令行与装配

- [ ] **T-15** `parse_args() -> Namespace`：位置参数 `INPUT` + 选项 `--backend` / `--force` / `--config` / `--pipeline-models` / `--vlm-models` / `--dry-run`（契约见 `contracts/cli.md`）
- [ ] **T-16** `process_one(pdf, args, env) -> str`：单文档全流程，返回 `"parsed"` / `"skipped"`
- [ ] **T-17** `print_report(dist) -> None`：按 `contracts/cli.md` §5 的固定格式打印类型分布报告
- [ ] **T-18** `main() -> int`：装配全流程；批量时**任一失败即返回该失败码**，不因后续成功而回退为 0

## Phase 5：依赖清单

- [ ] **T-19** 仓库根创建 `requirements.txt`：写 `mineru>=3.2.0,<4.0` + 注释说明版本号需安装后回填锁定；注明 CPU 版 PyTorch 的安装提示

## Phase 6：自检（不执行脚本的前提下）

- [ ] **T-20** 行数核对：`backend/parse_pdf.py` 落在 250–350 行
- [ ] **T-21** constitution 自检：无裸 `python` 调用；无硬编码密钥；函数签名带类型注解；标识符 `snake_case`；无吞异常的 `except`
- [ ] **T-22** 契约核对：退出码、输出格式、副作用清单与 `contracts/cli.md` 一致

---

## 依赖关系

```text
T-01..T-03  →  T-04..T-18  →  T-20..T-22
                    ↑
T-19（独立，可并行）
```

## 不在本特性范围内

- S3 清洗、S4 分块、S5 向量化、S6 入库（`docs/05 §5` 关键设计 2）
- manifest 准入闸门（用户明确要求省略，见 `plan.md` Complexity Tracking）
- 自动化测试框架（验证方式见 [quickstart.md](./quickstart.md)）
- 修改 `docs/` 下任何文档（回填清单见 `quickstart.md` 末尾）

## 执行前置条件

**用户已明确要求：只生成脚本，不执行。** 因此 T-11 的实跑验证、T-20 之后的实测回填，均不在此次执行范围内。
