# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本脚本职责：命令行离线建索引 —— 解析 PDF → 语义分块 → 向量化写入 Milvus → 落 BM25 语料。
# 用法：python scripts/build_index.py --pdf "data/uploads/招股说明书1.pdf" --no-mineru

import sys                             # 调整模块搜索路径，保证能 import app 包
import time                            # 统计各阶段耗时
import argparse                        # 命令行参数解析
from pathlib import Path               # 路径处理

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 把项目根目录加入 sys.path

from app import config as C            # 全局配置
from app import rag                    # 解析与语义分块
from app import store                  # 向量库与 BM25 存储


def main() -> int:                                                        # 脚本主函数
    """执行一次完整的建索引流程，返回进程退出码（0 成功，非 0 失败）。"""
    ap = argparse.ArgumentParser(description="为 PDF 建立向量与 BM25 索引")   # 参数解析器
    ap.add_argument("--pdf", default=str(C.UPLOAD_DIR / "招股说明书1.pdf"),
                    help="待索引的 PDF 路径")                               # 输入文件
    ap.add_argument("--no-mineru", action="store_true", help="跳过 MinerU，直接用 pymupdf 快速解析")  # 降级开关
    ap.add_argument("--drop", action="store_true", help="重建 collection（会清空现有全部向量）")   # 重建开关
    args = ap.parse_args()                                                # 解析命令行

    pdf = Path(args.pdf)                                                  # 转 Path
    if not pdf.exists():                                                  # 文件不存在
        print(f"[错误] 找不到 PDF：{pdf}")                                  # 提示
        return 2                                                          # 退出码 2

    print(f"[1/4] 开始解析：{pdf.name}（MinerU={'关闭' if args.no_mineru else '开启'}）")   # 进度
    t0 = time.time()                                                      # 计时
    blocks, parser = rag.parse_pdf(pdf, use_mineru=not args.no_mineru)    # 解析（内部含降级）
    print(f"      解析器={parser}，块数={len(blocks)}，耗时={time.time() - t0:.1f}s")   # 打印结果

    print("[2/4] 语义分块中（调用 bge-m3 计算句间相似度）…")                  # 进度
    t1 = time.time()                                                      # 计时
    chunks = rag.semantic_chunk(blocks)                                   # 语义分块
    tables = sum(1 for c in chunks if c["type"] == "table")               # 统计表格块
    print(f"      分块数={len(chunks)}（其中表格 {tables} 块），耗时={time.time() - t1:.1f}s")   # 打印

    print("[3/4] 向量化并写入 Milvus…")                                     # 进度
    t2 = time.time()                                                      # 计时
    doc_id = rag.make_doc_id(pdf)                                         # 内容指纹作为文档标识
    n = store.index_chunks(chunks, doc_id, pdf.name)                        # 入库 + BM25 落盘
    print(f"      写入 {n} 条，doc_id={doc_id}，耗时={time.time() - t2:.1f}s")   # 打印

    print("[4/4] 完成。知识库现有文档：")                                    # 进度
    for d in store.list_documents():                                        # 列出已入库文档
        print(f"      {d['doc_name']}  分块={d['chunks']}  最大页码={d['pages']}  表格={d['tables']}")   # 明细
    print(f"总耗时 {time.time() - t0:.1f}s")                               # 总耗时
    return 0                                                              # 成功退出码


if __name__ == "__main__":            # 仅在直接运行时执行
    raise SystemExit(main())          # 用返回值作为退出码
