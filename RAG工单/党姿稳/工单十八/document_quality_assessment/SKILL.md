---
name: DocumentQualityAssessmentSkill
description: 知识库文档质量评估技能。在构建知识库前对文档集合做体检：统计格式分布、把 PDF 分成文字型/扫描型/混合型以决定是否需要 OCR、给出文档长度分布、用 MD5 精确去重与 SimHash 近似去重找出重复与版本冲突、检测敏感信息，并为每份文档打上分类标签，供入库工作流路由到不同解析器。
version: 1.0.0
entry_point: document_quality_assessment.run_assessment
---

# DocumentQualityAssessmentSkill

## 这个技能解决什么问题

知识库里混着文字型 PDF、扫描件、Word、Markdown，直接丢给解析器会出现「扫描件解析出零字符」
「同一份文件入库多次」「正文里夹带手机号身份证」等问题。本技能在**入库前**做一次统一体检，
把问题文档挑出来，并给出每份文档该走哪个解析器的结论。

## 何时触发

- 用户要求评估/体检一个文档文件夹或知识库目录；
- 入库前需要判断哪些 PDF 是扫描件、需要先 OCR；
- 需要找重复文件、版本冲突、敏感信息；
- 需要为文档集合生成分类标签并决定解析路由。

## 工作流程（执行步骤）

1. **遍历与格式统计**：按配置的后缀清单遍历目标（支持文件夹路径、文件列表、单个文件），
   统计各类文件的数量、占比与体积。
2. **PDF 页面类型分流**：逐页判断，单页可提取字符数低于 `char_threshold` 记为扫描页；
   扫描页占比 ≥ `scan_ratio_threshold` 判为扫描型，≤ `text_ratio_threshold` 判为文字型，
   中间为混合型。此外用「图像覆盖率」做独立交叉验证，两者结论不一致或占比贴近阈值时，
   该文件进入 **待确认列表** 并给出一句话原因。
3. **文档长度分布**：统计全量文档字符数的 P25/P50/P75/P90/P99 与长度区间分布，并按格式细分。
4. **重复检测**：
   - MD5 精确匹配 → 完全相同文件，可直接确认；
   - SimHash 近似匹配（64 位指纹 + 分桶筛候选 + 汉明距离 ≤ `hamming_threshold`）→ 只能给出
     候选，输出 **待确认的版本冲突列表**，每条附带双方路径、相似度、汉明距离与文本片段。
5. **敏感信息检测**：手机号、邮箱、身份证（默认开启），银行卡默认关闭。
   身份证做校验位验证、命中值做排除规则过滤以压低误报。输出 **待审核列表**，
   每条含类型、命中值、字符偏移与上下文。
6. **分类标签**：主标签（Scan_PDF / Text_PDF / Mixed_PDF / DOCX / Markdown / …）决定解析路由；
   质量标签（Duplicate_Exact / Duplicate_Near / Sensitive_Info / Too_Short / Empty_Text / Corrupted）
   是可多选的问题标记。
7. **产出报告**：结构化 JSON + 可直接阅读的 HTML 简报。

## 资源层

| 资源 | 用途 |
|---|---|
| `assessment_config.yaml`（工单根目录） | 全部阈值与开关，改配置即可调整口径，代码无硬编码 |
| `document_quality_assessment/` | 技能实现（遍历、分诊、长度、去重、敏感、分类、报告） |
| `document_ingestion_workflow/` | 质检 → 决策 → 按标签路由到解析器的工作流 |
| `api_server.py` | `POST /v1/document/quality-inspection` 质检 API 端点 |

## 调用方式

```python
from document_quality_assessment import run_assessment
report = run_assessment(target=r"D:\知识库\documents", output_dir="data/report")
print(report["to_review"]["version_conflicts"])   # 待确认的版本冲突列表
print(report["to_review"]["sensitive_items"])     # 待审核的敏感信息列表
```

命令行：

```bash
python run_quality_check.py --target "D:\知识库\documents" --out data/report
```

## 注意事项

- 页面类型判定是启发式，不保证严格正确，所以阈值可配置且输出「待确认」列表；
- 扫描型 PDF 没有文本层，敏感信息与 SimHash 都需要文本，因此这类文档**必须先 OCR**
  （由 `document_ingestion_workflow` 路由到 `ocr_parser`）后再做检测；
- 近似重复只是候选，**必须人工确认**，技能不会自动删除任何文件；
- 处理大目录时结果按 `progress_every` 增量落盘到状态文件，中断后重跑会跳过已处理文件。
