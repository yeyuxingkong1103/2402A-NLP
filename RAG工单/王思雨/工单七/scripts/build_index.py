# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 入库入口脚本（核心逻辑在 app/ingest.py，此文件只做启动）
import sys, pathlib                                            # 让脚本能 import 到 app 包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 把项目根加进模块搜索路径
from app.ingest import build                                    # 导入入库主流程
if __name__ == "__main__":                                      # 直接运行本文件即执行入库
    n = build()                                                 # 解析全部 PDF → 语义分块 → 写 Milvus + BM25
    print(f"入库完成，共 {n} 个分块")                             # 打印块数，便于人工确认规模
