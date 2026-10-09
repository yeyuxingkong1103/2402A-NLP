---
name: DocumentQualityAssessmentSkill
description: 对知识库文档进行质量评估，输出格式分布、PDF类型、长度分布、重复检测、敏感信息与分类标签，用于RAG入库前的质检分流。
version: 1.0.0
author: 八维文化与产业研究院
tags: [rag, quality, pdf, ocr, duplicate, sensitive]
---

# Document Quality Assessment Skill

## 何时使用
- 构建知识库前，对原始文档目录做质量体检
- 需要判断 PDF 是文字型 / 扫描型 / 混合型
- 需要识别重复文件、版本冲突、敏感信息
- 需要为后续解析器路由提供分类标签

## 输入
- `folder_path`: 待评估文件夹路径（必填）
- `file_list`: 文件列表（可选，与 folder_path 二选一）
- `config_path`: 配置文件路径，默认 assessment_config.yaml

## 输出
结构化 JSON 报告，字段包括：
- `summary`: 总体统计
- `format_distribution`: 格式分布
- `pdf_classification`: PDF 类型分流
- `length_distribution`: 长度分位数与区间
- `duplicate_report`: MD5 精确重复 + SimHash 待确认冲突
- `sensitive_report`: 敏感信息待审核列表
- `classification_labels`: 文档分类标签
- `to_confirm`: 所有"待确认/待审核"列表汇总

## 工作流程
1. 遍历目录，收集文件元信息
2. 格式分布统计
3. PDF 逐页字符数判断，输出文字型/扫描型/混合型
4. 文档长度统计，输出 P25/P50/P75/P90/P99
5. MD5 精确去重；SimHash 相似度检测（可配置开关）
6. 敏感信息正则扫描，输出带上下文的待审核列表
7. 根据规则打分类标签：Clean_Markdown / Scan_PDF / Table_Heavy / Image_Heavy / Parse_Failed
8. 汇总报告，返回 JSON / HTML 简报

## 三层结构对应
- 元数据层：本文件 YAML frontmatter
- 指令层：本文件正文
- 资源层：core/ utils/ main.py assessment_config.yaml scripts/ tests/

## 资源
- 配置文件：assessment_config.yaml
- 运行脚本：scripts/run_assessment.py
- 单元测试：tests/
