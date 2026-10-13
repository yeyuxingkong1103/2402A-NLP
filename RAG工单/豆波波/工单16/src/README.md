# IMDR VLM fine-tuning

Convert QA records with `convert_imdr_to_jsonl.py`. The generated records contain `image`, `question`, `answer`, and standard `messages` fields. `lora_qwen_vl.yaml` is a LLaMA-Factory-style LoRA configuration. Evaluate held-out answers with `evaluate_industrial_vlm.py` for exact, terminology, and diagram-reasoning scores.
