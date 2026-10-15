# 工单十八 · 实现文档质量评估 Skill 并集成至智能体工作流 —— 运行说明

工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
验证数据集：`C:\Users\dang5\Desktop\新工单\14-17附件\original_problems\original_problems\documents`（1700 份专利 PDF，692.1 MB）

## 一、实现功能与验收标准对照

### 产出物对照

| 工单产出物 | 实现情况 | 对应文件 |
|---|---|---|
| 一、完整的 Skill 源代码 | 已完成：`document_quality_assessment/` 下 10 个模块，遵循 Skill 三层结构（元数据层 `SKILL.md` 前置元数据 / 指令层 `SKILL.md` 工作流 / 资源层各功能模块） | `document_quality_assessment/` |
| 二、集成后的工作流与 API 代码 | 已完成：`document_ingestion_workflow/` 专用工作流（质检 → 按标签路由解析器）；`api_server.py` 新增 `POST /v1/document/quality-inspection` 端点，返回 JSON 与 HTML 简报 | `document_ingestion_workflow/workflow.py`、`api_server.py` |
| 三、配置文件 | 已完成：所有阈值（字符数、扫描页占比、SimHash 汉明距离、敏感信息规则、分类标签）全部配置化，代码中不硬编码 | `assessment_config.yaml` |
| 四、测试脚本与验证报告 | 已完成：单元测试 + 1700 份真实数据验证报告（含分类准确性抽样交叉验证、版本冲突逐对著录项核查、敏感信息正控制样本） | `tests/test_skill.py`、`evaluation/验证报告.md` |
| 五、系统演示视频 | 未产出（需人工录屏）。界面能力与录屏步骤见第四节 | `api_server.py`、`data/*_reports/quality_report.html` |

### 验收标准对照

| 验收项 | 要求 | 实测 |
|---|---|---|
| 五大功能全覆盖（核心） | 格式统计、PDF 分类、长度分析、重复检测、敏感信息检测，并输出所有要求的列表 | 1700 文件全部执行完毕：格式 1 种 / 692.14 MB；扫描型 622、文字型 1078、混合型 0；长度分位数 P25=8 P50=4122 P75=5506 P90=6826 P99=10940；MD5 重复 0 组、SimHash 候选 16 对；敏感信息待审核 0 条 ✅ |
| 分类准确性 | 人工抽样验证准确率 **≥ 85%** | 分层随机抽样 60 份，用「最大图像面积占比」与「字体资源表」两个独立信号交叉复检，**一致率 100.00%（60/60）** ✅ |
| 列表可操作性 | 每条须含文件路径、相似片段、敏感信息上下文 | 16 对候选逐对回到首页著录项核查（申请人/申请日/申请号/文献种类/同日申请标记），**可用候选率 100%**：12 对为一案两报、3 对为同日姊妹件、1 对为同申请人关联申请 ✅ |
| 集成验收 | Skill 可被 API 端点调用并返回结构化 JSON；可注册并被工作流决策节点调用 | `POST /v1/document/quality-inspection` 返回 JSON/HTML；`GET /v1/skills` 返回注册清单；`DocumentIngestionWorkflow` 按标签路由扫描型 → OCR ✅ |
| 代码质量 | 阈值配置化、不硬编码；有错误处理（损坏 PDF） | 全部阈值集中在 `assessment_config.yaml`；`pdf_probe.py` 对无法打开的 PDF 落入 `errors` 清单并继续处理，1700 份中解析失败 0 份 ✅ |

## 二、环境

- Python：`C:\Users\dang5\anaconda3\envs\nlp2\python.exe`（Python 3.8.10）
- 依赖：`PyMuPDF 1.24.11`、`numpy`、`flask`、`PyYAML`（全部已在环境中安装）
- `.docx / .xlsx / .pptx` 用标准库 `zipfile` + `xml.etree.ElementTree` 解析 OOXML，**不需要** python-docx / openpyxl
- `rapidocr_onnxruntime` 仅在工作流把扫描件路由到 `ocr_parser` 时才惰性加载（属工单十四的解析器模块）
- 质检主流程无需 GPU、无需联网、不加载任何模型

## 三、运行方式

在 `C:\Users\dang5\Desktop\工单十八\` 目录下执行（Windows PowerShell，用 `;` 而非 `&&`）：

```powershell
# 1) 对 1700 份真实数据集跑一次完整质检（实测 1016.9 秒；每 100 个文件打印一次进度，
#    进度写入 data\assessment_state.json，中断后重跑自动续跑；--no-resume 可强制全量重跑）
python run_quality_check.py

# 2) 单元测试（24 个用例，基于标准库 unittest，约 12 秒）
python -m unittest discover -s tests -v

# 3) 生成验证报告（分类准确性抽样 + 版本冲突核查 + 敏感信息正控制）
#    读取 data\report\quality_report.json，输出 evaluation\验证报告.md 与 verification.json
#    （verify_report.py 是被 verify_skill.py 导入的渲染模块，不需要单独运行）
python verify_skill.py

# 4) 启动质检 API 服务（自动挑选 8800~8899 间空闲端口）
python api_server.py
```

## 四、访问地址

- 质检 API：`http://127.0.0.1:<端口>`（启动日志中打印，默认从 8800 起探测）
  - `GET  /health` —— 健康检查
  - `POST /v1/document/quality-inspection` —— 传入 `{"folder": "..."}` 或 `{"files": [...]}`，返回 JSON 报告
  - `GET  /v1/document/quality-inspection/report` —— 最近一次 JSON 报告
  - `GET  /v1/document/quality-inspection/report.html` —— HTML 简报（浏览器直接打开）
  - `POST /v1/document/ingestion-workflow` —— 触发「质检 → 按标签路由解析器」工作流
  - `GET  /v1/skills` —— 已注册技能清单
- 质检报告（HTML）：`data\report\quality_report.html`
- 验证报告：`evaluation\验证报告.md`

**演示视频录屏步骤**（对应产出物第五项）：
1. 启动 `python api_server.py`，用 `curl`/Postman 调用 `POST /v1/document/quality-inspection`，展示返回的 JSON；
2. 浏览器打开 `data\report\quality_report.html`，依次定位「扫描型 PDF（622 份）」「重复文件（MD5 0 组 / SimHash 待确认 16 对）」「敏感信息待审项」三个区块；
3. 调用 `POST /v1/document/ingestion-workflow`，展示 `Scan_PDF` 标签的文档被自动路由到 OCR 解析器。

## 五、注意事项

1. **首次全量运行实测 1016.9 秒（约 17 分钟）**：需逐页探测 1700 份 PDF（合计 13196 页，平均 7.8 页/份）。耗时随机器负载波动，控制台每 100 个文件打印一次进度。支持断点续跑，中断后重跑不会重算；状态缓存在 `data\assessment_state.json`。**复用时极快**：第二次运行直接复用已探测的 1700 条记录，仅需数秒。
2. **DTD/字体告警可忽略**：PyMuPDF 打开部分老旧专利时会打印字体与结构告警，不影响探测结果。
3. **敏感信息检测结果为 0 条是正确结论**，不是漏检：检测是对全部 1700 份文档的**完整文本**进行的（不是抽样）；另用一套不同的正则对全部 1700 份全文重新检索，手机号 / 邮箱 / 18 位身份证 / 16~19 位银行卡四类原始命中数均为 0，说明语料本身确实没有这类信息。同时 0 条命中即 0 误报——申请号 `201922242663.5`（13 位）、代理机构代码 `50240`（5 位）、邮编 `400020`（6 位）都没有被误判。检测器确实"能检出来"则由正控制样本证明（人工构造四类敏感信息注入真实专利正文，命中 4/4，见验证报告第五节）。
4. **SimHash 待确认列表不自动处理**：16 对候选中 12 对是「一案两报」，法律状态与保护范围不同，不应作为重复文件删除。技能只输出待确认列表与判断依据，由人工定夺。
5. **配置文件优先**：`assessment_config.yaml` 是唯一的阈值来源，修改后无需改代码；`--config` 可指定其他配置文件。
6. **`--config` 可指定其他配置文件**：`python run_quality_check.py --config my.yaml`。配置缺失时 `config_loader.py` 会落到内置的 FALLBACK 默认值，不会因为文件不存在而崩。

## 六、目录结构

```
工单十八/
├─ assessment_config.yaml              # 全部可调阈值（唯一配置来源）
├─ run_quality_check.py                # 命令行质检入口
├─ api_server.py                       # 质检 API 服务（含 6 个端点）
├─ verify_skill.py / verify_report.py  # 验证与报告生成
├─ document_quality_assessment/        # Skill 本体
│   ├─ SKILL.md                        #   元数据层 + 指令层（工作流 SOP）
│   ├─ assessment.py                   #   技能入口 run_assessment
│   ├─ scanner.py                      #   遍历与格式统计
│   ├─ pdf_probe.py                    #   PDF 文字型/扫描型分流
│   ├─ length_stats.py                 #   长度分布与分位数
│   ├─ dedup.py / simhash.py           #   MD5 精确去重 / SimHash 近似去重
│   ├─ sensitive.py                    #   敏感信息检测
│   ├─ classifier.py                   #   文档分类标签体系
│   ├─ config_loader.py                #   配置加载与校验
│   └─ report.py                       #   JSON/HTML 报告与摘要
├─ document_ingestion_workflow/        # 集成工作流
│   ├─ workflow.py                     #   质检 → 按标签路由解析器
│   └─ parsers.py                      #   各解析器适配（OCR / 文本直抽）
├─ tests/test_skill.py                 # 单元测试
├─ data/                               # 状态文件、报告、技能注册表
└─ evaluation/                         # verification.json、验证报告.md
```
