---
name: document-quality-assessment
description: 在构建特定领域知识库前，对各类知识库文件做质量评估。当用户需要"检查文档质量""知识库体检""扫描件识别""重复文件清理""敏感信息排查"，或要为 RAG 入库前的文档做分流（文字型走文本解析 / 扫描型走 OCR）时使用。输入是一个文件夹路径，输出结构化 JSON 报告 + HTML 简报，包含格式分布、PDF 页面类型识别、长度分布、重复检测、敏感信息检测和文档分类标签六大项。
version: 1.0.0
agent_created: true
---

# DocumentQualityAssessmentSkill（文档质量评估技能）

## 一句话说明

给一个文件夹，Skill 会输出一份「知识库体检报告」：有哪些格式、多少 PDF 是扫描件、文档长短分布、有没有重复文件、有没有敏感信息，并给每份文件打上**处理路由标签**（走文本解析还是走 OCR）。

## 三层结构（Anthropic Agent Skills 规范）

| 层 | 内容 | 加载时机 |
|---|---|---|
| 元数据层 | 就是本文件的 `name` + `description` | 始终加载（极低 token） |
| 指令层 | 本 SKILL.md 正文 | 触发时加载（中等 token） |
| 资源层 | `scripts/` 下的可执行脚本、`assessment_config.yaml` | 执行时才调用（token 大） |

## 何时使用

- 知识库入库前的质量体检
- 判断一批 PDF 里有多少需要 OCR
- 清理重复文档 / 发现疑似版本冲突
- 排查文档里是否混入了手机号、邮箱、身份证

## 怎么用

### 方式一：命令行（Skill 资源层脚本）

```bash
cd document_quality_assessment
python scripts/assess.py --input D:/data/imdr/documents --out ./产出物
```

常用参数：
- `--input` **必填**，待评估文件夹
- `--out` 报告输出目录（默认 `产出物/`）
- `--config` 配置文件路径（默认 `assessment_config.yaml`）
- `--limit N` 只处理前 N 个文件（调试用）
- `--resume` 断点续跑（1700 份文档耗时长，中断后可从断点继续）

产出：`quality_report.json` / `quality_report.html` / `quality_report.md`

### 方式二：HTTP API（已集成到 RAGFlow）

```bash
curl -X POST http://<host>/v1/document/quality-inspection \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"path":"/data/documents","limit":200}'
```

### 方式三：智能体工作流节点

工作流 `document_ingestion_workflow` 会把质检结果按标签路由：
`Scan_PDF → OCR 解析器`，`Structured_Table → 表格解析器`，`Long_Doc → 父子分块`。

## 六大功能与判定口径

| 功能 | 判定口径 | 输出 |
|---|---|---|
| 1 格式分布统计 | 按扩展名 + 按类别（文档/PDF/表格/演示/文本/图片）计数与占比 | `一_格式分布统计` |
| 2 PDF 页面类型 | **按页**：每页字符数 < 阈值 → 记扫描页；扫描页占比 ≥70% → `scanned`；≥15% → `mixed`；否则 `text` | `二_PDF页面类型识别` + **待确认列表** |
| 3 长度分布 | 统计每份文档字符数，输出 P25/P50/P75/P90/P99 + 6 个区间分布 | `三_文档长度分布` |
| 4 重复检测 | **两层**：MD5 精确匹配（可直接清理）+ SimHash 相似度 ≥0.90（**待确认的版本冲突列表**，带相似片段） | `四_重复检测` |
| 5 敏感信息检测 | 手机号 / 邮箱 / 身份证 默认开；**银行卡默认关**（误报率高）。输出**待审核列表**，每条带上下文 | `五_敏感信息检测` |
| 6 文档分类标签 | 处理路由标签（Scan_PDF/Mixed_PDF/Text_PDF/Structured_Table/Long_Doc/Empty_Content）+ 风险标签（Has_Sensitive/Exact_Duplicate/Version_Conflict） | `六_文档分类标签` |

> 关键设计：凡是需要人工判断的结论，一律输出**"待确认 / 待审核"列表并附带上下文**，
> 而不是只给一个统计数字——数字无法支撑操作员做决定。

## 标签 → 解析路由

| 标签 | 路由 |
|---|---|
| `Scan_PDF` | OCR 解析器（PaddleOCR / DeepDoc-OCR） |
| `Structured_Table` | 表格解析器（RAGFlowExcelParser） |
| `Long_Doc` | 父子分块（Parent-Child）+ Raptor |
| `Empty_Content` | 人工检查 / 换解析器 |
| 其余 | 默认文本解析器（naive） |

## 配置说明（`assessment_config.yaml`）

所有阈值都在配置文件里，**代码中不硬编码**：

```yaml
pdf:
  min_chars_per_page: 50     # 低于此字符数的页算"扫描页"
  scan_ratio_threshold: 0.70 # 扫描页占比超过此值判为扫描型
  mixed_ratio_threshold: 0.15
  max_pages_per_pdf: 40      # 每个 PDF 最多抽多少页（1700 份时的性能保护）
length:
  long_doc_threshold: 50000
dedup:
  enable_simhash: true
  simhash_threshold: 0.90
sensitive:
  enable: {手机号: true, 邮箱: true, 身份证: true, 银行卡: false}
limits:
  max_files: null            # null=不限；设数字可只跑前 N 个
```

## 性能提示

处理 1700 份文档耗时较长：
1. 先用 `--limit 100` 试跑确认配置；
2. 正式跑加 `--resume`，中断后从断点继续（每个模块结束都会落盘 `_checkpoint.json`）；
3. `pdf.max_pages_per_pdf` 控制每份 PDF 的抽样页数，是全量跑的主要性能开关。

## 已验证结果（IMDR 数据集 60 份样���）

- 文件总数 60、总大小 40.52 MB
- **扫描型 PDF 60 / 60（100%）**，混合型 0，文字型 0
- 需 OCR 路由：60 份 → 与 IMDR「不可编辑的图片型 PDF」数据集特征一致
- MD5 精确重复 0 组、SimHash 版本冲突 0 组、敏感信息待审核 0 条
- 长度分布：全部为 0 字符（扫描件无可提取文本，需 OCR 后才能统计真实长度）
