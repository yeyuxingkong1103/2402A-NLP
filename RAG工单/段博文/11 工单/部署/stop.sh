#!/bin/bash
# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
# 停止微调/编码进程
pkill -f "python finetune.py" 2>/dev/null && echo "已停止微调进程" || echo "无运行中的微调进程"
