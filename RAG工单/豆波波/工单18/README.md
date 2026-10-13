# 工单18 - 文档质量评估 Skill 与 Agent 工作流

目录包含完整的 `document_quality_assessment` 包、`SKILL.md`、CLI/工作流、API 参考和离线样例。它支持格式统计、PDF 文本/扫描/混合分类、长度分位数、MD5 重复、SimHash 版本冲突和敏感信息初筛。

```powershell
python run_quality_assessment.py --input 规格说明.pdf --output quality_report.json
```

接入 RAGFlow API 时使用 `POST /v1/document/quality-inspection`；生产环境仍需人工复核敏感信息和 OCR 结论。

