# 工单十八 · 实现文档质量评估 Skill 并集成至智能体工作流

为知识库入库前提供文档质检能力：格式统计、PDF 文字型/扫描型分流、长度分布、MD5+SimHash 去重、敏感信息检测，输出“待确认/待审核”列表，并集成到智能体工作流自动路由（扫描型→OCR）。

## 目录
```
document_quality_assessment/
  SKILL.md                      Skill 定义（元数据/指令/资源 三层）
  assessment_config.yaml        全部阈值配置
  src/                          评估引擎、SimHash、报告渲染
  workflow/                     document_ingestion_workflow（智能体工作流）
  api/                          质检 API（POST /v1/document/quality-inspection）
  tests/                        单元测试
scripts/                        数据集生成、运行、图表、验证脚本
docs/                           架构/接口/验证报告、流程图、思维导图、11 张截图
deploy/                         install.sh / start.sh / stop.sh
output/                         质检报告 quality_report.json / .html
```

## 快速开始
```bash
cd 工单十八
conda activate rag18           # 或 bash deploy/install.sh
# 1) 构造模拟数据集并运行质检
bash scripts/run_all.sh
# 2) 单元测试
cd document_quality_assessment && python tests/test_quality_assessor.py
# 3) 启动 API
bash deploy/start.sh 8028
curl -X POST http://127.0.0.1:8028/v1/document/quality-inspection \
  -H 'Content-Type: application/json' -d '{"path":"demo_dataset"}'
```

## 验收对照
- 五大功能全覆盖，阈值全配置化，损坏 PDF 有错误处理；
- 列表可操作（待确认 PDF / 版本冲突 / 敏感信息均含路径与上下文）；
- Skill 可经 API 调用、可被工作流决策节点路由（Scan_PDF→OCR）。

详见 `docs/01_测试验证报告.md`。
