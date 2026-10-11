# DocumentQualityAssessmentSkill 设计文档

> 工单：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流（工单 18，V1.1-20260123）

---

## 1. 技能定位与三层结构

按 Anthropic **Agent Skills** 规范，本 Skill 分三层：

| 层 | 内容 | 加载时机 | Token |
|---|---|---|---|
| 元数据层 | `SKILL.md` frontmatter：name / description | 始终 | 极低 |
| 指令层 | `SKILL.md` 正文：工作流程 SOP | 触发时 | 中 |
| 资源层 | `config.py` `detectors.py` `report.py` `workflow.py` `api.py` `assessment_config.yaml` | 执行时按需 | 大 |

## 2. 五大基本功能设计

### 2.1 格式分布统计
遍历文件夹（按 `extensions` 过滤），统计各扩展名**数量与占比**，输出 `format_stats.by_ext`。
→ `detectors.format_stats()`

### 2.2 PDF 页面类型识别（启发式分流）
- **逐页判定**：用 PyMuPDF 取每页文本长度，`< pdf_min_chars_per_page` 记为**扫描页**；
- **整篇判定**：扫描页占比 `> scan_page_ratio(0.70)` → **扫描型**；文字页占比 `> text_page_ratio` → **文字型**；否则 **混合型**；
- **可配置 + 待确认**：所有阈值在 yaml；损坏/空文档归入 `pdf_types.待确认`。
→ `detectors.pdf_page_type()`

> 说明：这是**启发式**（用于分流与估算），不保证严格正确，故输出「待确认」列表供人工复核。

### 2.3 文档长度分布
统计全部文档**字符数**，输出 `P25/P50/P75/P90/P99` 与**长度区间分布**（<1k / 1k–5k / 5k–20k / 20k–100k / >100k）。
→ `detectors.length_distribution()`

### 2.4 重复检测（两层机制）
- **MD5**（精确）：分块哈希文件字节 → 完全相同文件分组；
- **SimHash**（近似）：对正文做字符 shingle(3) → 64 位签名 → 汉明距离 `≤3` 判为高相似 → 输出 **「待确认的版本冲突列表」**（`file_a/file_b/hamming/status`）。
- ⚠️ 对**无文本层的扫描型 PDF**，正文为空 → 不参与 SimHash（报告已注明）。
→ `detectors.md5_duplicates()` / `detectors.near_duplicates()`

### 2.5 敏感信息检测
正则检测 **手机号 / 邮箱 / 身份证**；**银行卡默认关闭**（误报率高）。
关键改进：从「只给统计数字」变为 **「展示待审核列表」**——每条命中附 **上下文**（前后各 `context_chars`）。
→ `detectors.sensitive_scan()`

## 3. 文档分类标签体系

标签由**证据**组合生成（`detectors.classify_doc`）：

| 标签 | 触发条件 | 用途 |
|---|---|---|
| `Scan_PDF` | PDF 判为扫描型 | 路由到 **OCR 解析器** |
| `Text_PDF` | PDF 判为文字型 | 直接文本解析 |
| `Mixed_PDF` | PDF 判为混合型 | OCR + 文本 |
| `Patent` | 文件名含「专利」或 `CN\d+` | 走 DeepDOC |
| `ShortDoc` / `LongDoc` | 字符数 <200 / ≥200 | 评估是否适合固定分块 |
| `Markdown`/`PlainText`/`Word`/`Excel`/`Slides`/`Other` | 非 PDF 按扩展名 | 选择对应解析器 |

## 4. 集成设计

### 4.1 工作流 `document_ingestion_workflow`
```
触发质检 → DocumentQualityAssessmentSkill.assess() → 按分类标签路由：
  Scan_PDF → ocr_parser
  Mixed_PDF → hybrid_parser
  Text_PDF → plain_parser
  Patent → deepdoc_parser
```
→ `workflow.document_ingestion_workflow()`

### 4.2 API 端点
`POST /v1/document/quality-inspection`
- 入参：`{"path": "<文件夹>"}` 或 `{"files": [...]}`；`format:"html"` 返回 HTML 简报；
- 出参：`{"code":0,"data":{...报告...},"html":"..."}`。
→ `api.run_inspection()` / `api.build_app()`

## 5. 健壮性与工程性
- **阈值全配置化**：`assessment_config.yaml` + `Config`；
- **错误处理**：损坏 PDF 捕获异常并归入「待确认」，不中断整批；
- **进度反馈 + 中断恢复**：`progress_cb` + `progress_file` 落盘，重跑跳过；
- **单测**：`tests/test_skill.py` 覆盖五大功能（7 项）。

## 6. IMDR 实测结论（见 03_演示与结果）

> 详见 `03_演示与结果/质检报告_IMDR.json|html` 与下方的抽样表。

### 6.1 分类标签人工抽样表（模板）

| 文件 | 自动标签 | 人工判定 | 是否一致 |
|---|---|---|---|
| 待填 | Scan_PDF | 待填 | 待填 |
| …（抽 30~50 份） | | | |
| **准确率** | | | **__%**（目标 ≥85%） |
