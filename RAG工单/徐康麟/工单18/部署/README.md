# 工单18 部署说明

**工单编号**：人工智能NLP-RAG-文档质量评估 Skill 工单

> 复现命令、API 位置、配置说明与实测数字，全部整理在
> `测试/测试报告与部署说明.md`（含一键复现命令与关键运维事实）。
> 本文件仅作部署入口指引，避免重复内容。

## 快速入口

| 用途 | 位置 |
| --- | --- |
| 一键复现命令 | `测试/测试报告与部署说明.md` §3 |
| 阈值配置 | `研发/assessment_config.yaml` |
| CLI | `& $py -m document_quality_assessment --help` |
| 全量报告 | `研发/data/reports/quality_report_full.json` / `.html` |
| 逐文件明细 | `研发/data/reports/per_file.jsonl`（1700 行） |
| 交付物核验 | `部署/交付物核验.md` |