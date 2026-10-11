---
name: document-quality-assessment
description: 对知识库文件夹做入库前质检：格式分布统计、PDF 文字型/扫描型分流、文档长度分布、MD5+SimHash 重复检测、敏感信息检测，并输出结构化 JSON 报告与 HTML 简报及分类标签。适用于构建领域知识库前的批量文档质量评估。
---

# DocumentQualityAssessmentSkill（文档质量评估技能）

## 何时使用
在**构建/扩充知识库之前**，需要对一批文档做质量体检时使用：
- 需要知道文件夹里各类格式的数量与占比；
- 需要把 PDF 分成「文字型 / 扫描型 / 混合型」，决定是否走 OCR；
- 需要文档长度分布（分位数），判断是否适合固定分块；
- 需要找出完全重复（MD5）与高相似版本（SimHash）文件；
- 需要排查文档中的敏感信息（手机号/邮箱/身份证；银行卡默认关闭）。

## 三层结构
- **元数据层（本 frontmatter）**：技能名与简介，始终加载，Token 消耗极低；
- **指令层（本 SKILL.md）**：工作流程（下述），触发时加载；
- **资源层（脚本/配置/模板）**：`config.py`、`detectors.py`、`report.py`、`assessment_config.yaml`、`workflow.py`、`api.py`，按需执行。

## 工作流程（SOP）
1. **加载配置**：`Config.load("<assessment_config.yaml>")`（所有阈值可配置，**不硬编码**）。
2. **执行评估**：`assess(root, cfg, limit=0, progress_file="progress.json")`
   - 遍历 `root`（按 `extensions` 过滤）；
   - 逐文件：PDF → 页面类型识别（每页字符数 < `pdf_min_chars_per_page` 记为扫描页）；文本 → 抽正文；
   - 汇总：格式分布、PDF 分流（含「待确认」列表）、长度分位数、MD5 重复、SimHash 近似重复（「待确认版本冲突」）、敏感信息（「待审核」列表）、分类标签。
   - **进度与中断恢复**：每 100 个文件落盘 `progress_file`，重跑跳过已完成项。
3. **生成报告**：`report.to_json()` / `report.to_html()`（HTML 简报突出「待确认 / 待审核」清单）。
4. **工作流路由**：`workflow.document_ingestion_workflow()` 按标签路由——`Scan_PDF→ocr_parser`、`Text_PDF→plain_parser`、`Mixed_PDF→hybrid_parser`、`Patent→deepdoc_parser`。
5. **对外接口**：`POST /v1/document/quality-inspection`（`api.py`）。

## 输出（结构化字段）
- `format_stats`：按扩展名的数量与占比；
- `pdf_types`：扫描型/文字型/混合型计数 + `待确认`；
- `length_distribution`：P25/P50/P75/P90/P99 + 区间分布；
- `duplicates.md5`：完全重复分组；`duplicates.near`：SimHash 待确认；
- `sensitive`：命中统计与 `待审核` 明细（含上下文）；
- `classification.label_counts`：分类标签体系统计。

## 调用示例
```bash
python -m document_quality_assessment.cli --root <目录> \
  --config document_quality_assessment/assessment_config.yaml \
  --out report.json --html report.html
```
