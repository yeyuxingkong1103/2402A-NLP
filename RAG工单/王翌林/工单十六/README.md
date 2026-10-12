# 工单十六：调试与微调专用视觉语言模型（VLM）

> 人工智能 NLP-RAG 项目 · 工单十六 · 领域适应（Domain Adaptation）

## 1. 背景与目标

RAGFlow 默认 VLM（如 Qwen-VL）对**工业领域术语**（淬火、公差配合、气化器、电极……）和**复杂工程图纸**的理解不足，
导致检索到图文块后生成的答案专业度低、术语错误、图纸推理偏差。

本工单使用 **IMDR 工业专利语料**对基座 VLM 做 **LoRA 领域微调**，并给出**专业评估**，使模型：
- 在工业 QA 上输出更贴近标准答案（BLEU/ROUGE 提升）；
- 更准确地复述工业专业术语（术语准确性提升 ≥ 5%）；
- 形成可复用的「数据 → 微调 → 评估 → 部署」流水线。

## 2. 验收标准达成情况

| 验收项 | 结果 |
|--------|------|
| 微调能启动并完成 ≥ 1 epoch、损失正常下降 | ✅ 完成 3 epoch，loss 4.23 → 1.0~2.0，train_loss=2.14 |
| 微调后工业术语准确性较基线提升 ≥ 5% | ✅ 见 `docs/screenshots/` 评估对比 |
| 评估报告含失败案例分析与超参建议 | ✅ 见 `docs/评估与优化报告.md` |

## 3. 目录结构

```
工单十六/
├── src/
│   ├── finetune.py          # LoRA 微调主程序（transformers + peft）
│   └── evaluate.py          # 专业评估（BLEU/ROUGE + 术语准确 + 图纸推理）
├── configs/lora_sft.yaml    # 微调超参配置
├── scripts/
│   ├── build_finetune_data.py  # IMDR QA + 图文块 → JSONL
│   ├── run_finetune.sh / run_finetune3.sh
│   ├── run_eval.sh
│   ├── gen_diagrams.py / .sh  # 生成架构图/流程图/思维导图/曲线
│   ├── install.sh / start.sh / stop.sh   # 部署脚本
├── data/
│   ├── train.jsonl / eval.jsonl   # 102 / 18 条 SFT 样本
│   ├── images/<doc>/page_XX.png   # 专利页面渲染图
│   └── logs/                      # 训练/评估日志
├── finetune_output/
│   ├── lora_adapter/              # LoRA 适配器权重（adapter_model.safetensors）
│   └── train_history.json         # 训练 loss 曲线
└── docs/
    ├── screenshots/               # 设计图、训练曲线、评估结果截图（多张）
    └── *.md                       # 技术/测试/部署/评估文档
```

## 4. 快速开始

```bash
# 安装依赖（含 CUDA 检查）
bash scripts/install.sh

# 一键运行：数据 → 微调 → 评估 → 出图
bash scripts/start.sh

# 或分步
bash scripts/run_finetune3.sh     # 微调 3 epoch
bash scripts/run_eval.sh          # 评估对比
bash scripts/gen_diagrams.sh      # 生成截图
```

环境变量：
- `PYTHON_BIN`：Python 解释器（默认 venv）
- 基座模型默认位于 `/home/dabaie/models/Qwen2-VL-2B-Instruct`

## 5. 硬件与环境

- GPU：RTX 5060 Laptop 8GB
- 显存策略：fp16 + gradient checkpointing + 冻结视觉塔 + batch=1（4bit 量化可选）
- 框架：torch 2.11 + transformers + peft + accelerate

详见 `docs/部署文档.md`。
