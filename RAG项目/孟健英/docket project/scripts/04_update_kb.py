# -*- coding: utf-8 -*-
"""知识库动态更新：追加新的 PDF 或文本到 Milvus。

用法：
  python scripts/04_update_kb.py data/guidelines/new_content.txt
  python scripts/04_update_kb.py data/guidelines/another_pdf.pdf
"""
import sys  # sys.argv 取命令行参数：文件路径由外部传入，脚本不硬编码路径
from pathlib import Path  # 路径处理：从命令行参数建 Path 对象，判存在、读扩展名、读文本

from src.config import settings, setup_logging  # 配置中心 + 日志初始化
from src.ingestion import split_pages, chunk_pages, embed_and_insert  # 复用已有管线：切页/切块/入库
from src.pdf_parser import parse_and_clean  # PDF 解析：支持命令行直接传 PDF，无需先跑 02

setup_logging()  # 初始化日志：记录追加入库的文件与条数


def main():
    if len(sys.argv) < 2:  # 参数校验：用户忘了传文件时给出用法提示
        print("用法: python scripts/04_update_kb.py <文件路径>")  # 标准用法：告诉用户怎么传参
        print("示例: python scripts/04_update_kb.py data/guidelines/new_content.txt")  # 示例路径：降低用户输入成本
        return  # 终止：没有文件路径就无从处理

    file_path = Path(sys.argv[1])  # 用户传的路径转成 Path：自动处理分隔符
    if not file_path.exists():  # 判存在：友好提示避免底层 FileNotFoundError
        print(f"❌ 文件不存在: {file_path}")  # 提示用户检查路径
        return  # 终止：文件缺失没法继续

    print(f"⏳ 处理文件: {file_path}")  # 进度提示：开始处理

    # 根据扩展名选择解析方式
    if file_path.suffix.lower() == ".pdf":  # PDF 走 PyMuPDF 解析链路：页眉页脚剥离 + 表格提取
        text = parse_and_clean(file_path)  # 解析+清洗一条龙：产出带页码标记的文本
    elif file_path.suffix.lower() in (".txt", ".md"):  # 纯文本直接读入：markdown 和 txt 一视同仁
        text = file_path.read_text(encoding="utf-8")  # 纯文本不需要解析，直接读到内存
    else:  # 格式兜底：不支持的扩展名直接拒绝，避免把二进制或乱码送进向量库
        print(f"❌ 不支持的格式: {file_path.suffix}")  # 明确告知不支持
        return  # 终止：格式未知不硬猜

    print(f"   清洗后字数: {len(text)}")  # 文本规模：若字数异常少说明解析失败，用户应去查日志
    pages = split_pages(text) or [(1, text)]  # 无[第N页]标记按整篇一页处理
    chunks = chunk_pages(pages, source=file_path.name)  # 滑窗切块并附来源文件名
    print(f"   切块数: {len(chunks)}")  # 块数统计：追加量心里有数

    count = embed_and_insert(chunks)  # 向量化并入库 Milvus：检索和入库必须用同一个 Embedding 模型，否则向量空间不匹配——这里复用的是 ingestion 里的 get_embedder，保证与全量初始化一致
    print(f"✅ 增量更新完成，新增 {count} 条")  # 完成提示：入库条数回显，用户核对是否与块数一致


if __name__ == "__main__":
    main()  # 离线建库补充流程：不删旧库直接追加新文件，增量更新让知识库随文档版本持续生长
