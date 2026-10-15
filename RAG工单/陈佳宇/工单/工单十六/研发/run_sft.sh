# -*- coding: utf-8 -*-
#!/bin/bash
# run_sft.sh 工单16 LoRA微调启动脚本
source /mnt/c/Users/陈/Desktop/工单5/venv_gongdan5/bin/activate
cd /mnt/c/Users/陈/Desktop/工单16/研发

llamafactory‑cli sft --config vlm_lora_sft.yaml
