# RAG 工单 14-18 实现说明

本实现按附件工单要求落地了可运行的离线能力，并保持 RAGFlow 现有模块边界。

## 已实现

- 工单 14/15：PDF 文本/混合内容质检基础、PDF 解析器进程内单例缓存，降低重复加载 OCR/布局模型的开销。
- 工单 16：`scripts/vlm/convert_imdr_to_jsonl.py` 将 QA 与图片块转换为多模态 JSONL；`evaluate_industrial_vlm.py` 输出精确匹配、术语覆盖和图纸推理指标；`lora_qwen_vl.yaml` 提供 LoRA 训练配置。
- 工单 17：`observability.concurrency.RequestLimiter` 限制并发生成；`observability.metrics` 提供 Prometheus 文本指标；对话流式和非流式请求均接入限流与耗时统计；系统新增 `/v1/system/metrics`。
- 工单 18：`document_quality_assessment` Skill 实现格式统计、PDF 分流、长度 P25/P50/P75/P90/P99、MD5 精确重复、SimHash 版本冲突、手机号/邮箱/身份证敏感信息上下文审核；新增 `document_ingestion_workflow` 和 `POST /v1/document/quality-inspection`。

## 使用

```bash
PYTHONPATH=. python -m document_quality_assessment.cli /path/to/documents \
  --config document_quality_assessment/assessment_config.yaml \
  --json quality_report.json --html quality_report.html
```

VLM 数据转换：

```bash
python scripts/vlm/convert_imdr_to_jsonl.py questions.jsonl data/imdr_vlm.jsonl --image-root documents
```

## 验证

已通过 Python 编译检查、离线断言测试、VLM 转换/评估样例，以及对新工单 14-18 五份 PDF 的批量质检。完整 Docker、Redis、向量库、GPU 微调和 10/20 并发压测需要外部运行环境，当前环境未启动这些依赖。
