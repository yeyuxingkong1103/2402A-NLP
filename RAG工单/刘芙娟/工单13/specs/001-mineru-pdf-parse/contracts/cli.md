# CLI Contract: `backend/parse_pdf.py`

**Feature**: `specs/001-mineru-pdf-parse` | **Date**: 2026-09-22

本文件是脚本对外的**稳定契约**。实现可变，本契约不可随意变。

---

## 1. 调用方式

```text
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py [INPUT] [OPTIONS]
```

> constitution 原则 I：文档与示例中 MUST NOT 出现裸 `python` / `python3` / `py`。

---

## 2. 位置参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `INPUT` | 否 | `data/source_data/` | 单个 PDF 文件路径，或包含 PDF 的目录。目录时处理其中全部 `*.pdf` |

**默认值语义**：不传 `INPUT` 即处理 `data/source_data/`——用户后续把新 PDF 丢进该目录后，直接重跑即可。

---

## 3. 选项

| 选项 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `--backend` | `pipeline` \| `vlm-engine` | `pipeline` | MinerU 后端。`pipeline` 是 CPU 官方支持路线且与 `docs/04 §12` 一致 |
| `--force` | 标志 | 关 | 强制重新解析，忽略已有产物 |
| `--config` | 路径 | 脚本内常量 | 覆盖项目内 MinerU 配置文件的生成位置 |
| `--pipeline-models` | 路径 | 脚本内常量 | 覆盖 pipeline 权重目录（本机有两份，一份残缺） |
| `--vlm-models` | 路径 | 脚本内常量 | 覆盖 vlm 权重目录 |
| `--dry-run` | 标志 | 关 | 只打印将要执行的操作与解析出的 `doc_id`，不调用 MinerU |
| `-h` / `--help` | — | — | 帮助 |

**为什么 `--pipeline-models` / `--vlm-models` 要做成选项**：本机 `C:` 与 `E:` 各有一份 pipeline 权重，且 `E:` 那份缺 4 个子目录。做成命令行选项后，更换权重位置无需改代码（constitution N3：关键参数显式，不依赖默认值）。

---

## 4. 退出码

对齐 `docs/05 §5` 的既有约定：

| 码 | 含义 | 触发条件 |
|---|---|---|
| `0` | 成功 | 全部待处理文档已完成解析（含"全部命中跳过"的情形） |
| `1` | 参数错误 | `INPUT` 不存在 / 不是 PDF / `--backend` 取值非法 / 权重路径不存在 |
| `2` | 校验失败 | `content_list.json` 缺失或为空 / 存在无 `page_idx` 的块 / 出现未知 `type` |
| `3` | 外部依赖不可用 | `mineru` 未安装 / MinerU 子进程非 0 退出 |

**批量处理时的退出码**：任一文档失败即返回该失败的最高优先级码（`3` > `2` > `1`），**不因后续文档成功而回退为 0**。

---

## 5. 标准输出契约

成功时**必然**打印（顺序固定）：

```text
[1/2] doc_id=d6da41b5d356  file=国家基层高血压防治管理指南2025版.pdf
      sha256=d6da41b5d3565c49...
      status=parsed
      output=data/parsed/d6da41b5d356/

--- 类型分布 (backend=pipeline, mineru=3.x.y) ---
  text           380
  table           12
  page_number     15
  ...
  合计            452
  page_idx 范围   [0, 14]
  未知类型        无

[2/2] doc_id=xxxxxxxxxxxx  file=...
      status=skipped (产物已存在且完整)
```

**失败时**打印到标准错误，且**不产生**可被误认为成功的产物：

```text
ERROR: content_list.json 中第 42 个块缺少 page_idx 字段
ERROR: 未知的块类型 'foobar'（docs/04 §5 未覆盖）—— 停止并需人工决策
ERROR: 未找到 mineru 可执行文件：<路径>；请先按 requirements.txt 安装
ERROR: 权重目录不存在：<路径>；拒绝联网下载
```

---

## 6. 环境变量契约

脚本**读取**（可选，均为覆盖用）：

| 变量 | 作用 |
|---|---|
| `MED_RAG_REPO_ROOT` | 覆盖仓库根目录的自动推断（用于跨机器） |

脚本**写出**（仅注入 MinerU 子进程，不修改当前进程与系统）：

| 变量 | 值 |
|---|---|
| `MINERU_TOOLS_CONFIG_JSON` | 项目内配置文件的绝对路径 |
| `MINERU_MODEL_SOURCE` | `local` |
| `MINERU_DEVICE_MODE` | `cpu` |

---

## 7. 副作用清单

脚本运行**只允许**产生以下副作用：

1. 创建/覆盖 `data/parsed/{doc_id}/` 及其下 MinerU 产物
2. 创建/覆盖 `.mineru/mineru.json`
3. 打印到标准输出/标准错误

**明确禁止**的副作用：

- 修改 `C:\Users\Lenovo\mineru.json`（用户明确要求不碰全局配置）
- 修改系统环境变量
- 联网（模型权重必须已在本地，`docs/04 §11`）
- 写入 `data/source_data/`（输入目录只读）
- 修改 `requirements.txt` 或任何源码
