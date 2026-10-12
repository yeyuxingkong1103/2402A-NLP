---
name: DocumentQualityAssessmentSkill
description: 对知识库文档进行质量评估，输出格式分布、PDF类型识别、长度分布、重复检测、敏感信息检测和分类标签。
version: 1.0.0
triggers:
  - 文档质量
  - 质量评估
  - 质检
---

# DocumentQualityAssessmentSkill

对指定文件夹下的所有文档进行批量质量评估，为 RAGFlow 知识库入库做前置质检。

## 核心能力
1. 格式分布统计
2. PDF 页面类型识别（文字型/扫描型/混合型）
3. 文档长度分布（分位数 + 区间）
4. 重复检测（MD5 + SimHash）
5. 敏感信息检测（手机/邮箱/身份证/银行卡）
6. 文档分类标签（Clean_Markdown/Scan_PDF/Table_Heavy/Image_Heavy/Parse_Failed）
