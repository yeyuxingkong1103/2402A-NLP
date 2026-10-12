---
name: document-quality-assessment
description: 对知识库文件夹进行文档质量评估——格式分布统计、PDF 文字型/扫描型/混合型分流、文档长度分布、MD5+SimHash 重复检测、敏感信息检测，输出结构化报告与“待确认/待审核”列表。当需要对一批文档做入库前质检、去重、敏感信息排查时使用。
---

# Document Quality Assessment Skill

## 触发场景
构建/更新领域知识库前，对 `documents/` 等文件夹做入库质检，定位扫描型 PDF、重复文件、敏感信息等需人工介入的项。

## 工作流程（SOP）
1. **接收输入**：文件夹路径或文件列表 + `assessment_config.yaml`。
2. **格式分布统计**：遍历统计 `.pdf/.docx/.md` 等数量与占比。
3. **PDF 分流**：按页字符数判定扫描页，扫描页占比 > 阈值标记为 `Scan_PDF`，介于区间为 `Mixed_PDF`，输出“待确认”列表。
4. **长度分析**：计算 P25/P50/P75/P90/P99 与长度区间分布，标记超长/超短离群。
5. **重复检测**：MD5 精确去重；SimHash 找高相似文档，输出“待确认的版本冲突列表”（附相似片段）。
6. **敏感信息检测**：手机号/邮箱/身份证（银行卡默认关），输出“待审核列表”（命中值 + 上下文）。
7. **分类打标签 + 路由**：按 `classification.route_map` 给每个文档打标签并指定解析器（`Scan_PDF` → `ocr_parser`）。
8. **输出报告**：JSON 全文 + HTML 简报，重点呈现待确认/待审核项。

## 调用方式
```python
from src.quality_assessor import assess
report = assess("/path/to/documents")   # 返回 dict
```
或通过 API：`POST /v1/document/quality-inspection`。

## 资源层（按需加载）
- `assessment_config.yaml`：全部阈值配置（不得硬编码）。
- `src/`：评估引擎、SimHash、报告渲染。
- `report_renderer.py`：JSON→HTML 简报。

## 注意
- 启发式判定不保证 100% 准确，所有阈值可配置，结果以“待确认/待审核列表”交付人工复核。
- 损坏 PDF 有错误处理，不中断整体流程。
