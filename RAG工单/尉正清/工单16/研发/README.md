# 工单16 运行说明

**工单编号**：人工智能NLP-RAG-微调专用视觉语言模型
**环境**：conda `rag_gd2`（从 `rag_gd1` 克隆，装了 peft / bitsandbytes / accelerate）
**基座**：`D:/Pycharm/yzq/models/Qwen2-VL-2B-Instruct`

## 完整流程

```bash
conda activate rag_gd2
cd D:/Pycharm/yzq/zg6/工单16/研发

# 1. 数据转换：IMDR QA 对 + RAGFlow 图文块 → JSONL(image/question/answer)
#    --shuffle-options 只打乱训练集的选项顺序（评估集保持原样），
#    不加的话模型会去学「答案多半在 B」的位置先验，见 优化/过程问题记录.md 第 10 条
python prepare_data.py --out data --eval-docs 60 --max-train 600 --ragflow --shuffle-options

# 2. 评估基线（未微调）
python evaluate.py --tag base

# 3. LoRA 微调（配置在 finetune_config.json）
python finetune.py --config finetune_config.json

# 4. 评估微调后
python evaluate.py --tag finetuned --adapter output/qwen2vl-2b-lora-imdr

# 5. 出对比报告
python ../测试/compare.py
```

冒烟测试（不跑全量，验证链路）：

```bash
python finetune.py --limit 8 --max-steps 2
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 数据格式转换脚本（工单产出物 1） |
| `finetune_config.json` | 微调配置：模型 / LoRA / 训练超参 / 图像分辨率（工单产出物 2） |
| `finetune.py` | QLoRA 训练循环（自己写，不依赖 trl） |
| `evaluate.py` | 专业评估脚本（工单产出物 3） |
| `data/` | 转换出来的 JSONL 与渲染好的图纸 PNG |
| `output/` | 训练产出的 LoRA adapter |

## 关键取舍

**为什么 4-bit QLoRA**：GPU 只有 6GB（RTX 3060 Laptop）。Qwen2-VL-2B 的
bf16 权重就要 4.4GB，加上激活和优化器状态放不下；4-bit 量化后权重约 1.2GB。

**视觉塔必须一起训**（`lora.vision_lora=true`）：只训语言侧的话，模型学到的是
「答案多半在 B」的位置先验而不是读图 —— 把训练集选项打乱后增益立刻归零。
详见 `测试/评估报告.md` 第三节。

**为什么训练循环自己写**：这个项目的环境里深度学习库互相锁版本
（见工单11/14 的过程记录）。`trl` 会连带拉 transformers 的版本要求进来，
风险不值当；训练循环本身不复杂。

**图像分辨率上限 602112 像素**（`finetune_config.json` 的 `image.max_pixels`）：
Qwen2-VL 默认把图放到 `1280×28×28`，一张 A4 图纸就能吃掉 2GB 显存。
按 28 的整数倍给到约 784×768，识别图上标注足够，显存也扛得住。
