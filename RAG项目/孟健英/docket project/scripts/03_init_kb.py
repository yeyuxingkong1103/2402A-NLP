# -*- coding: utf-8 -*-
"""全量初始化知识库：读取清洗文本 → 切块 → 向量化 → 入库 Milvus。"""
from pathlib import Path  # 路径处理：拼接、换扩展名、判存在都用它

from src.config import settings, setup_logging  # 配置中心 + 日志初始化：离线建库脚本统一先配日志
from src.ingestion import split_pages, chunk_pages, embed_and_insert  # 入库管线三件套：切页→切块→向量化入库

setup_logging()  # 日志初始化：建库过程的耗时与异常写入文件，事后复盘有据可查


def main():
    txt_path = Path(settings.guideline_path).with_suffix(".txt")  # 由 PDF 路径推导出同名的 .txt：02_parse_guideline.py 的产物
    if not txt_path.exists():  # 前置依赖检查：先解析后入库，顺序不能跳
        print(f"❌ 找不到清洗文本: {txt_path}")  # 提示缺失：用户知道该回头跑 02
        print("   请先运行: python scripts/02_parse_guideline.py")  # 给出明确命令：降低用户试错成本
        return  # 终止：文本不存在时切块/向量化都会报错

    print(f"⏳ 读取文本: {txt_path}")  # 进度提示：大文件读取可能有几秒，先让用户知情
    text = txt_path.read_text(encoding="utf-8")  # 读入整篇清洗文本到内存：通常几十到几百 KB，可全量载入

    print("⏳ 切块中...")  # 进度提示：切块快但向量化慢
    pages = split_pages(text) or [(1, text)]  # 无[第N页]标记的纯文本按整篇一页处理
    chunks = chunk_pages(pages, source=txt_path.name)  # 按滑窗切块并附加来源文件名：chunk_pages 返回 [{text, source, page}, ...]
    print(f"   共 {len(chunks)} 个块")  # 块数预估：总条目 = 总页数 × 每页块数（非整除时会有余数），帮助用户核对规模

    embed_and_insert(chunks)  # 向量化 + 入库 Milvus：耗时大头的计算与网络 I/O 都在这一步
    print("🎉 知识库初始化完成！")  # 成功提示：离线建库流程走到终点


if __name__ == "__main__":
    main()  # 离线建库第二步：清洗文本 → 分块 → 向量入库；执行完后 Milvus 里就有可检索的知识了
