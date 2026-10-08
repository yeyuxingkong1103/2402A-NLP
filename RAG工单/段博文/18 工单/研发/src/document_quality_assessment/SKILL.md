---
name: document-quality-assessment
description: >-
  知识库入库前的文档质量评估技能。当需要在构建 RAG/领域知识库之前，对一个文件夹或文件列表做
  质量质检时使用。支持格式分布统计、PDF 文字型/扫描型/混合型分流、文档长度分布、MD5 精确
  去重与 SimHash 近似版本冲突检测、手机号/邮箱/身份证等敏感信息检测，并为每个文档生成分类
  标签与解析路由建议，输出结构化 JSON 报告与 HTML 简报。所有阈值均可通过
  assessment_config.yaml 配置，启发式判断结果会进入"待确认/待审核"列表。
---

# 文档质量评估 Skill（DocumentQualityAssessmentSkill）

## 用途
在文档进入知识库解析流水线**之前**完成质检，回答三个问题：
1. 这批文档整体质量如何（格式、长度、损坏、扫描件占比）？
2. 哪些文档不能直接处理，需要 OCR、人工确认或安全审核？
3. 每个文档应该被路由到哪个解析器？

## 输入
- 一个文件夹路径（默认递归扫描），或
- 一个显式文件列表。
- 可选：配置文件路径（缺省使用本目录 `assessment_config.yaml`）。

## 执行流程（SOP）
按以下顺序执行，不要跳步：

1. **加载配置**：读取 `assessment_config.yaml`，所有阈值只从配置取，不硬编码。
2. **收集文件**：按 `scan.extensions` 过滤；无法识别后缀的文件记入 `unsupported_files`。
3. **逐文件评估**（有进度反馈，结果写 checkpoint 支持中断恢复）：
   - 计算 size、MD5；
   - 用 `text_reader` 统一提取文本（pdf/docx/md/txt）；打不开的损坏文件标记 `Corrupt_File`，不中断整体流程；
   - PDF 按页启发式分类：单页字符数 < `pdf.text_char_threshold` 为扫描页；
     扫描页占比 > `scanned_file_ratio` 标记 `Scan_PDF`，介于 `mixed_min_ratio` 与该值之间标记 `Hybrid_PDF`，否则 `Text_PDF`；
   - 统计字符数；
   - 敏感信息检测（手机/邮箱/身份证默认开，银行卡默认关），每条命中**必须带上下文**，进入"待审核列表"；
   - 生成分类标签与路由建议（见下）。
4. **汇总分析**：
   - 格式数量与占比；
   - 长度分位数 P25/P50/P75/P90/P99 与区间分布；
   - MD5 完全重复分组；
   - SimHash 两两汉明距离比较（长度粗筛提速），距离 ≤ 阈值的进入"待确认版本冲突列表"，**必须给出相似片段**。
5. **输出报告**：结构化 JSON（机器读）+ HTML 简报（人工读），顶部突出三类待办：
   扫描型待 OCR、版本冲突待确认、敏感信息待审核。

## 分类标签与路由
| 标签 | 含义 | 路由 |
| --- | --- | --- |
| `Text_PDF` / `DOCX_Doc` / `Markdown_Doc` / `Text_Doc` | 可直接提取文本 | DirectParser |
| `Scan_PDF` | 扫描型，需 OCR | OCRParser |
| `Hybrid_PDF` | 混合型 | HybridParser（文本层 + 扫描页 OCR） |
| `Empty_Doc` | 空/近空文档 | 人工审核 |
| `Corrupt_File` | 文件损坏无法打开 | 人工审核 |
| `Exact_Duplicate` | MD5 完全重复 | 保留一份，其余跳过 |
| `Near_Duplicate` | SimHash 高相似 | 人工确认版本冲突 |
| `Sensitive_Info` | 命中敏感信息 | 安全审核后再入库 |

## 重要约束
- PDF 页面分类是**启发式**，用于分流与成本估算，不保证严格正确；所有阈值可配置，
  分类结果附置信依据（扫描页占比），临界样本（扫描页占比 60%~80%）自动进入"待确认"列表。
- MD5 是确定结论；SimHash 只给嫌疑，**不自动删除任何文件**。
- 损坏文件、权限不足文件只报错记录，不允许让整个评估崩溃。

## 配套资源
- `assessment_config.yaml`：全部可调参数。
- `skill.py`：主编排入口 `DocumentQualityAssessment`。
- `format_stats.py / pdf_classifier.py / length_analysis.py / duplicate_detector.py / sensitive_detector.py / doc_classifier.py`：五大功能与标签。
- `text_reader.py`：统一文本读取；`report.py`：JSON/HTML 报告；`workflow.py`：智能体工作流。
