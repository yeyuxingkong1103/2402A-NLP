# Phase 1 Data Model: MinerU PDF 解析产物

**Feature**: `specs/001-mineru-pdf-parse` | **Date**: 2026-09-22

本文件定义解析步骤的**产物契约**。字段名以 `docs/04_数据管线设计.md` §2 与 §5 为唯一权威来源，本文件不新增字段，只把它们精确化为可实现的形式。

---

## 1. 实体关系

```text
源文档 (SourceDocument)          1 PDF 文件
   │  内容哈希
   ▼
文档标识 (doc_id)                sha256[:12]
   │  解析
   ▼
解析产物 (ParsedArtifact)        data/parsed/{doc_id}/
   │  含
   ▼
内容块 (ContentBlock) [0..N]     content_list.json 的数组元素，每项带 page_idx
   │  统计
   ▼
类型分布报告 (TypeDistribution)  运行期产出，不落盘
```

---

## 2. 源文档 (SourceDocument)

| 属性 | 类型 | 来源 | 约束 |
|---|---|---|---|
| `path` | `Path` | 命令行参数或默认目录扫描 | 必须存在、可读、`*.pdf` |
| `sha256` | `str` | 文件内容计算 | 64 位十六进制小写 |
| `file_name` | `str` | `path.name` | 原样保留，供引用卡片展示 |

**验证规则**：
- 路径不存在 → 退出码 `1`，不创建任何产物目录
- 文件不可读 / 非 PDF / 加密 → 由 MinerU 子进程报错，脚本映射为退出码 `3`
- 目录扫描时，同名文件只处理一次（按 `sha256` 去重）

---

## 3. 文档标识 (doc_id)

| 属性 | 类型 | 约束 |
|---|---|---|
| `doc_id` | `str` | `sha256[:12]`，小写十六进制，长度恒为 12 |

**语义**（`docs/04 §10.1`）：**内容一变，`doc_id` 即变**。这是"同名不同内容不得混入"的机制。

**验证规则**：`doc_id` 由脚本计算，**不从任何外部文件读取**（本特性不实现 manifest，见 `plan.md` 的 Complexity Tracking）。

---

## 4. 解析产物 (ParsedArtifact)

落盘位置：`data/parsed/{doc_id}/`

| 文件 | 是否进下游管线 | 说明 |
|---|---|---|
| `content_list.json` | ✅ **唯一消费对象** | 扁平、按阅读顺序，每项带 `page_idx` |
| `middle.json` | ❌ | 页级中间结构，供页码复核与调试 |
| `*.md` | ❌ | 人类可读，仅作人工抽查 |
| `layout.pdf` | ❌ | 版面可视化，人工排查解析错误用 |

**为什么必须是 `content_list.json` 而非 Markdown**（`docs/04 §5`）：MinerU 的 Markdown 输出**不携带页码信息**。一旦基于 Markdown 分块，页码永久丢失，而引用卡片依赖页码。这是本步骤最容易踩的坑。

> ⚠️ 实际文件名与层级由 MinerU 版本决定，脚本须**容错定位**（见 `tasks.md` T-08），并在定位失败时报错退出而非静默跳过。

---

## 5. 内容块 (ContentBlock)

`content_list.json` 数组的每个元素。字段依后端而异，以下为 `docs/04 §5` 记录的 **3.x pipeline 后端**契约。

### 5.1 公共字段

| 字段 | 类型 | 约束 |
|---|---|---|
| `type` | `str` | 块类型，取值集合见 §5.5 |
| `page_idx` | `int` | **0-based 页级索引**；物理页码 = `page_idx + 1`。**必须存在**，缺失即报错退出 |
| `bbox` | `list[int]` | 0–1000 归一化整数，`[x0, y0, x1, y1]`。**office 后端无此字段** |

### 5.2 正文与标题（`type == "text"`）

| 字段 | 类型 | 约束 |
|---|---|---|
| `text` | `str` | 正文内容 |
| `text_level` | `int` | **仅标题块存在此键**，值即层级（1 = 一级标题） |

> ⚠️ **最容易写错的一点**（`docs/04 §5`）：
> - 标题判定是 `type == "text"` **且 `"text_level"` 键存在**；
> - **不存在 `type == "title"` 这种类型**；
> - 正文中该键是**完全缺失**，不是 `0`——因此**不能**用 `item.get("text_level", 0) >= 1` 判断。
> - pipeline 后端层级**无上限**；VLM 后端**上限为 4**。

### 5.3 表格（`type == "table"`）

| 字段 | 类型 | 约束 |
|---|---|---|
| `table_body` | `str` | **HTML 字符串**，可能缺失（此时回退用 `img_path`） |
| `table_caption` | `list[str]` | **与表体分离**，需显式拼回 |
| `table_footnote` | `list[str]` | **与表体分离**，需显式拼回 |
| `img_path` | `str` | 表格图像相对路径 |

### 5.4 图片与公式

| 类型 | 字段 | 约束 |
|---|---|---|
| `image` | `image_caption` / `image_footnote` | 均为 `list[str]`。⚠️ **2.0.0 叫 `img_caption`，2.1.0 起改名** |
| `equation` | `text` / `text_format` | `text_format` 恒为 `"latex"`，公式在 `text` 字段 |

### 5.5 `type` 取值集合（3.x pipeline 后端）

| 取值 | 含义 | `docs/04 §6` 的处置 |
|---|---|---|
| `text` | 正文或标题（凭 `text_level` 键区分） | 保留 |
| `image` | 图片 | 保留 |
| `table` | 表格 | 保留 |
| `chart` | 图表 | — |
| `equation` | 公式 | 保留 |
| `code` | 代码块 | — |
| `list` | 列表；`sub_type == "ref_text"` 为参考文献条目 | `ref_text` 丢弃，`text` 保留 |
| `header` / `footer` | 页眉 / 页脚 | 丢弃 |
| `page_number` | 印刷页码 | 丢弃 |
| `aside_text` | 侧边装饰文字 | 丢弃 |
| `page_footnote` | 页脚注释 | **默认保留**（可能是剂量注解） |
| `discarded` | 解析器主动丢弃的块 | 丢弃，全文记入日志 |

> ⚠️ **未知类型处理**（`docs/04 §5`）：出现本表未列出的类型 → **停止并人工决策，不得静默忽略**。未知类型意味着可能有内容被无声跳过。本特性将未知类型计入退出码 `2`。

---

## 6. 类型分布报告 (TypeDistribution)

运行期产出，**不落盘**，仅打印到标准输出。用于核对 MinerU 实际输出契约是否与 `docs/04 §5` 的假设一致（对应 `docs/04 §15` 待验证项 P1）。

**结构**：

```jsonc
{
  "mineru_version": "3.x.y",          // 实测回填的版本号
  "backend": "pipeline",
  "doc_id": "d6da41b5d356",
  "total_blocks": 452,
  "type_counts": { "text": 380, "table": 12, "page_number": 15, ... },
  "page_idx_range": [0, 14],           // 期望等于 [0, 页数-1]
  "unknown_types": []                  // 非空 → 退出码 2
}
```

**验证规则**：
| 检查项 | 失败阈值 | 处置 |
|---|---|---|
| `page_idx` 缺失的块数 | `> 0` | 退出码 `2`，指出具体块索引 |
| `unknown_types` 非空 | 非空 | 退出码 `2`，列出未知类型名 |
| `page_idx_range` 上界与 PDF 实际页数不符 | 不符 | 告警（不阻断），指出可能的页码语义偏差 |
| `total_blocks == 0` | 等于 0 | 退出码 `2`，**不得**写出空产物冒充成功 |

---

## 7. 项目内 MinerU 配置 (`.mineru/mineru.json`)

由脚本生成，用于覆盖全局配置。**只写模型路径与来源，不写任何凭据**。

```jsonc
{
  "models-dir": {
    "pipeline": "<C: 缓存中 PDF-Extract-Kit-1.0 的 master 目录绝对路径>",
    "vlm":      "<C: 缓存中 MinerU2.5-Pro-2605-1.2B 的 master 目录绝对路径>"
  },
  "model-source": "local",
  "config_version": "1.3.2"
}
```

**约束**：
- `model-source` 必须为 `local`——设为 `modelscope` 会在缺模型时**联网下载**，违反 `docs/04 §11`
- **不复制**全局配置中的 `bucket_info`（含 ak/sk/endpoint）与 `llm-aided-config`（含 api_key），即便其值当前只是占位符（constitution 原则 III）
- `vlm` 与 `pipeline` 路径**不可共用**同目录（MinerU 要求）

---

## 8. 环境变量（脚本内设置，不修改系统环境）

| 变量 | 值 | 作用 |
|---|---|---|
| `MINERU_TOOLS_CONFIG_JSON` | 项目内配置文件的**绝对路径** | 绕过 `~/mineru.json` 全局配置 |
| `MINERU_MODEL_SOURCE` | `local` | 强制离线，禁止联网兜底 |
| `MINERU_DEVICE_MODE` | `cpu` | 本机无 NVIDIA 显卡；`-d` 参数自 3.0.0 已删除 |

这些变量只注入到 MinerU 子进程的 `env`，**不写回当前进程环境，也不修改系统环境变量**。
