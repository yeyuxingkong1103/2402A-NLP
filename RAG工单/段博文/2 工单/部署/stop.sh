#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 结束脚本（Linux）：停止 RAG-PDF 问答服务
pkill -f "uvicorn main:app" 2>/dev/null && echo "服务已停止" || echo "未发现运行中的服务"
