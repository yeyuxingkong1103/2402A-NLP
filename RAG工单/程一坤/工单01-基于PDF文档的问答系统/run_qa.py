# -*- coding: utf-8 -*-
"""
命令行问答入口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
功能：终端交互式问答，展示答案 + 引用页码 + 耗时。
运行：python run_qa.py
"""
import sys
import os
# 把上级目录的"00-公共模块"加入模块搜索路径，以便 import rag_engine / ollama_client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00-公共模块"))

# 导入 RAG 引擎：封装 检索（向量召回TopK）+ 提示词拼装 + LLM 生成 的完整问答流水线
from rag_engine import RAGEngine
# 导入 Ollama 客户端单例 client：用于检测本地 Ollama 服务是否在线
from ollama_client import client


def main():
    # 先检查 Ollama 服务是否存活，避免后续 embedding/生成请求报连接错误
    if not client.is_alive():
        print("[错误] Ollama 服务未启动，请先运行: ollama serve")
        return

    try:
        # 加载工单01构建好的索引 zgs1_v1（加载 .npy 向量矩阵 + .json 分块元数据）
        engine = RAGEngine("zgs1_v1")
    except RuntimeError as e:
        # 索引不存在时 RAGEngine 会抛 RuntimeError，这里给出友好提示并退出
        print(f"[错误] {e}")
        return

    print("=" * 60)
    print("基于PDF文档的问答系统（工单01）| 输入 q 退出")
    print("知识库: 招股说明书1.pdf")
    print("=" * 60)

    # 主循环：不断读取用户问题并回答，直到输入 q/quit/exit 退出
    while True:
        # input() 阻塞读取一行，strip() 去掉首尾空白
        q = input("\n你的问题 > ").strip()
        # 空输入直接进入下一轮，不触发检索
        if not q:
            continue
        # 输入 q/quit/exit（不区分大小写）时退出循环
        if q.lower() in ("q", "quit", "exit"):
            break
        try:
            # 执行完整 RAG 问答：with_context=True 表示检索到上下文后交给 LLM 生成
            r = engine.ask(q, with_context=True)
        except Exception as e:
            # 单个问题失败（如 Ollama 超时）不应中断整个会话，打印后继续
            print(f"[出错] {e}")
            continue
        # 打印 LLM 生成的答案
        print(f"\n【回答】\n{r['answer']}")
        # 打印引用来源：每个检索命中的分块对应页码与相似度分数
        print(f"\n【引用来源】" + ", ".join(f"第{s['page']}页(相似度{s['score']})" for s in r["sources"]))
        # 打印分项耗时：检索耗时 + 生成耗时 = 总耗时（验收标准 3 秒内）
        print(f"【耗时】检索 {r['retrieval_time']}s + 生成 {r['generation_time']}s = {r['time_cost']}s")


if __name__ == "__main__":
    main()
