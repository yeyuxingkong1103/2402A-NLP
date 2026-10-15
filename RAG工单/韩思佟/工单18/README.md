# 工单18 · 实现文档质量评估 Skill 并集成至智能体工作流

工单编号：人工智能NLP-RAG-实现文档质量评估 Skill 并集成至智能体工作流

## 一、工单目标

实现 DocumentQualityAssessmentSkill（格式分布/PDF类型/长度分布/MD5+SimHash去重/敏感信息/分类标签），注册为智能体工具，新增质检 API，并在 IMDR 数据集上验证。

## 二、怎么跑

```bash
python 工单18/document_quality_assessment/scripts/assess.py --input data/imdr/sample60 --out 工单18/产出物
python 工单18/document_quality_assessment/tests/test_dqa.py
```

## 三、产出物
- `工单18/document_quality_assessment/SKILL.md`
- `工单18/document_quality_assessment/assessment_config.yaml`
- `工单18/产出物/quality_report.html`
- `工单18/产出物/quality_report.json`
- `工单18/集成/document_ingestion_workflow.json`

## 五、需要截图的地方

见同目录 `截图清单.md`。