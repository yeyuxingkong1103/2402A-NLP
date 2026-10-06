# 工单 18：实现文档质量评估 Skill 并集成至智能体工作流

**工单编号**：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单

## 一、背景：MCP 与 Skills

- **MCP**：对“工具调用”的协议标准化（解决 N 模型 × M 工具的复杂度），只定义“手怎么动”；
- **Skills**：对“业务逻辑编排”的标准化（定义“脑怎么想”），本质是标准化的 SOP，
  封装任务定义 + 执行流程 + 配套资源。

Skill 三层结构：

| 层 | 加载时机 | 内容 | Token |
|----|----------|------|-------|
| 元数据层 | 始终加载 | 技能名称与简介 | 极低 |
| 指令层 | 触发时加载 | `SKILL.md` 核心工作流程 | 中等 |
| 资源层 | 按需加载 | 脚本、参考文档、模板数据 | 大 |

## 二、DocumentQualityAssessmentSkill 功能

| 功能 | 说明 |
|------|------|
| 格式分布统计 | 遍历文件夹，统计 `.pdf/.docx/.md` 等数量与占比 |
| PDF 页面类型识别 | 按页字符数 < 阈值判扫描页，扫描页占比 > 70% 标记扫描型；阈值可配置，输出“待确认”列表 |
| 文档长度分布 | 字符数分位数 P25/P50/P75/P90/P99 + 长度区间分布 |
| 重复检测 | MD5 精确匹配 + SimHash 高相似（输出“待确认的版本冲突列表”） |
| 敏感信息检测 | 手机号/邮箱/身份证（银行卡默认关闭），每条记录附上下文 |

## 三、目录结构

```
18_工单/
├── document_quality_assessment/
│   ├── SKILL.md               # 元数据层 + 指令层
│   ├── __init__.py
│   ├── assessor.py            # 核心质检逻辑（资源层脚本）
│   └── assessment_config.yaml # 所有可调阈值配置
├── workflow.py                # document_ingestion_workflow（质检→路由）
├── api_server.py              # POST /v1/document/quality-inspection（JSON+HTML）
├── test_assessor.py           # 单元测试
├── main.py                    # 在 IMDR 数据集上运行
├── config.py
├── requirements.txt
└── README.md
```

## 四、集成与运行

```bash
pip install -r requirements.txt
python test_assessor.py                     # 单元测试
python main.py [文件夹路径]                  # 在真实数据集上运行
python api_server.py                        # 启动质检 API
# 调用：
curl -X POST http://localhost:8001/v1/document/quality-inspection \
     -H "Content-Type: application/json" -d '{"folder": "..."}'
curl -X POST ".../quality-inspection?format=html" ...   # 返回 HTML 简报
```

智能体工作流：`workflow.py` 中 `document_ingestion_workflow` 完成
`触发质检 → 调用 Skill → 按分类标签路由解析器`（Scan_PDF → OCR 解析器）。

## 五、配置说明（assessment_config.yaml）

所有阈值配置化：`scan_page_char_threshold`、`scanned_ratio_threshold`、
`length_percentiles`、`simhash_hamming_threshold`、`detect_bankcard`、`context_window` 等。

## 六、验收对照

- 五大功能全覆盖，输出所有要求列表（待确认版本冲突、待审核敏感信息等）；
- 分类标签（Text_PDF/Scan_PDF/Mixed_PDF/Duplicate/…）抽样准确率 ≥85%；
- 列表可操作：每项含文件路径、相似片段、敏感信息上下文；
- API 端点成功调用并返回结构化 JSON（含 HTML 简报）；
- Skill 可注册并被工作流决策节点调用；
- 阈值配置化、有损坏 PDF 错误处理；
- 进度反馈与中断恢复（`main.py` 落盘 `quality_report.json`）；
- 代码注释含工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单。
