# 专用视觉语言模型（VLM）微调方案

> 工单：人工智能NLP-RAG-调试与微调专用视觉语言模型（工单 16，V1.1-20260123）
> 目标：让 RAGFlow 集成的 VLM 适应**工业术语**（淬火、公差配合…）与**复杂图纸**理解。

---

## 1. 为什么默认 VLM 在工业场景表现差

- RAGFlow 默认 VLM（Qwen-VL 等）多在通用图文语料训练，**工业术语与图纸符号（尺寸线、部件编号、装配关系）分布不同**，存在领域鸿沟；
- IMDR 文档为**低分辨率图片型 PDF**：图纸中**部件编号小、线条密**，对模型分辨率与细节感知要求高；
- 结果：术语答非所问、图纸推理靠常识乱猜。

## 2. 数据准备（`01_代码/convert_to_vlm_dataset.py`）

- **来源 1（QA 对）**：IMDR `questions.jsonl`（10096 题，含选项与答案、group 分组）；
- **来源 2（图文块）**：RAGFlow 解析出的图文块（图 + 图注/描述）——可由工单 14 的 DeepDOC/VLM 解析输出接入；
- **转换逻辑**：
  1. 从问题文本解析**页码引用**（`第N页`），用 PyMuPDF 把该页**放大渲染成 PNG**（默认 zoom=2，低分辨率图纸放大以利 OCR/视觉）；
  2. 用户侧 = `问题 + 选项`，前面加 `<image>` 占位；回答侧 = **正确选项全文**（含字母）；
  3. 输出 LLaMA-Factory VLM 格式：`{"messages":[...],"images":[...],"meta":{...}}`。
- **实测**（本地已跑）：在 `CN100342976C.pdf`/`CN100347506C.pdf` 上生成样例 → `03_演示与结果/vlm_dataset_sample.jsonl`（含 `images/*.png`）。

## 3. 微调配置（`01_代码/lora_config.yaml`，LLaMA-Factory）

| 项 | 取值 | 说明 |
|---|---|---|
| 基座 | `Qwen/Qwen2.5-VL-7B-Instruct` | 与 RAGFlow 默认 VLM 同族 |
| 微调方式 | **LoRA**，rank=8，alpha=16，dropout=0.05 | 低秩起步，省显存 |
| `lora_target` | all（语言塔） | 视觉塔 `freeze_vision_tower: true` |
| 图像预处理 | `image_max_pixels=262144` / `min=200704` | 控制显存，兼顾图纸细节 |
| 文本 | `template=qwen2_vl`，`cutoff_len=2048` | Qwen-VL 专用模板 |
| 训练 | lr **1e-4**、batch 2×grad-accum 8（等效 16）、**2 epoch**、cosine、bf16、grad checkpoint | |
| 验证 | `val_size=0.02`，每 200 步评估 | |

启动：`llamafactory-cli train lora_config.yaml`（见 `01_代码/run_train.sh`）。

## 4. 评估设计（`01_代码/eval_industrial.py`）

除 **BLEU（字符 4-gram）/ ROUGE-L** 外，设计两项**工业专项**指标：

1. **专业术语准确性**：对每题标注的术语（如 `部件4/部件5`、`配气带孔盘`、`P·吉特勒`），统计 pred 的**术语覆盖率**；
2. **图纸推理正确性**：对 `type=image` 的样本，用**选项命中判定**（预测字母 vs 金标准字母）衡量「看图推理」是否与图纸相符。

输出：总准确率、文本题准确率、图文题准确率、术语覆盖率、BLEU/ROUGE（JSON 落盘）。

## 5. 验收对照与后续

| 验收项 | 方案 |
|---|---|
| ≥1 epoch、loss 正常下降 | 2 epoch，`plot_loss` 出曲线 |
| 术语准确率提升 ≥5% | 对比「基线 VLM」与「微调后」的 `term_accuracy` |
| 报告含失败案例 + 建议 | 见 `03_演示与结果/微调前后评估报告.md` |

**下一步超参/数据建议**（实跑后据结果细化）：
- 若欠拟合：提升 `lora_rank`(8→16)、lr 或 epoch；
- 若过拟合：加 dropout、扩数据、早停；
- 数据清洗：剔除渲染模糊页（低分辨率放大后仍不可读的）、统一术语表。
