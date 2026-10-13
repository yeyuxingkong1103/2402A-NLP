# 工单16 - VLM 调试与微调

目录包含 `convert_imdr_to_jsonl.py`、`evaluate_industrial_vlm.py`、`lora_qwen_vl.yaml` 和离线评估入口。转换、指标计算可本地运行；LoRA 训练需要 GPU 与对应训练框架。

```powershell
python run_offline_eval.py --input sample_imdr.json --output vlm_eval.json
```

