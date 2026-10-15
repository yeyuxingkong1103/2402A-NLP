# 工单16 · 调试与微调专用视觉语言模型

工单编号：人工智能NLP-RAG-调试与微调专用视觉语言模型

## 一、工单目标

准备 VLM 微调数据（JSONL: image/question/answer），配置并启动 LoRA 微调，用 BLEU/ROUGE + 专业术语准确性 + 图纸推理正确性评估，微调后优于基线。

## 二、怎么跑

```bash
python 工单16/code/build_vlm_dataset.py
llamafactory-cli train 工单16/code/qwen2vl_lora_sft_imdr_2b_qlora.yaml
python 工单16/code/evaluate_vlm.py --backend api --model <base> --tag base
python 工单16/code/evaluate_vlm.py --backend api --model <merged> --tag ft
```

## 三、产出物
- `工单16/code/启动命令.md`
- `工单16/code/qwen2vl_lora_sft_imdr.yaml`
- `工单16/code/evaluate_vlm.py`
- `工单16/产出物/数据集统计.json`
- `工单16/产出物/评估_base.json`
- `工单16/产出物/评估_ft.json`

## 五、需要截图的地方

见同目录 `截图清单.md`。