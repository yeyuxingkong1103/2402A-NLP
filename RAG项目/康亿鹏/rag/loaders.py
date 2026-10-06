"""文档加载与切分。"""  # 模块说明
from pathlib import Path  # 路径处理工具
from typing import List  # 类型注解：列表

from langchain_community.document_loaders import (  # LangChain 社区提供的各类文档加载器
    CSVLoader,  # CSV 加载器
    Docx2txtLoader,  # Word (.docx) 加载器
    PyPDFLoader,  # PDF 加载器（按页加载）
    TextLoader,  # 纯文本加载器
)

from langchain_core.documents import Document  # LangChain 文档对象（page_content + metadata）
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符切分器

import config  # 全局配置（切分参数在这里）

# 后缀 -> Loader（.md 按纯文本处理，避免引入 unstructured 重依赖）
LOADER_CLASSES = {  # 文件后缀到加载器类的映射表
    ".txt": TextLoader,  # 纯文本文件
    ".md": TextLoader,  # Markdown 当作纯文本处理
    ".pdf": PyPDFLoader,  # PDF 文件
    ".docx": Docx2txtLoader,  # Word 文档
    ".csv": CSVLoader,  # CSV 表格
}

SUPPORTED_EXTS = set(LOADER_CLASSES)  # 支持的文件后缀集合 {'.txt', '.md', '.pdf', '.docx', '.csv'}

# 保留到 metadata 的字段白名单（其余一律丢弃，避免不同加载器/PDF 元数据字段不一致导致 Milvus schema 冲突）
METADATA_KEEP = {"source"}


def build_loader(path: Path):  # 根据文件后缀构造对应的加载器实例
    loader_cls = LOADER_CLASSES[path.suffix.lower()]  # 查映射表得到加载器类（后缀统一转小写）
    if loader_cls in (TextLoader, CSVLoader):  # 文本类加载器需要处理编码问题
        # 自动检测编码，兼容 GBK / UTF-8
        return loader_cls(str(path), encoding="utf-8", autodetect_encoding=True)  # 开启编码自动探测
    return loader_cls(str(path))  # 其他类型（pdf/docx）直接构造，无需编码参数


def load_directory(docs_dir: str) -> List[Document]:  # 递归加载目录下所有文档
    """递归加载目录下所有支持的文档。"""
    docs: List[Document] = []  # 结果容器
    for path in sorted(Path(docs_dir).rglob("*")):  # 递归遍历所有路径（排序保证加载顺序稳定）*是匹配任意名字
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:  # 只处理支持的文档文件
            try:  # 单个文件失败不影响整体
                loaded:List[Document] = build_loader(path).load()  # 实例化加载器并加载内容
                for doc in loaded:  # 清洗 metadata：只保留白名单字段，丢弃 PDF 元数据（comments/producer/creator 等）
                    doc.metadata = {k: v for k, v in doc.metadata.items() if k in METADATA_KEEP}
                docs.extend(loaded)  # 累加到结果列表
                print(f"[加载] {path} -> {len(loaded)} 页")  # 打印加载进度
            except Exception as exc:  # 捕获任意加载异常
                print(f"[跳过] {path}: {exc}")  # 打印原因并跳过该文件
    return docs  # 返回全部加载的文档


def split_documents(docs: List[Document]) -> List[Document]:  # 把长文档切分成片段
    """按字符长度切分，针对中文补充分隔符。"""
    splitter = RecursiveCharacterTextSplitter(  # 构造递归切分器
        chunk_size=config.chunk_size,  # 每个片段最大字符数
        chunk_overlap=config.chunk_overlap,  # 相邻片段重叠字符数
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],  # 分隔符优先级：空行>换行>中文标点>空格>逐字符
    )
    return splitter.split_documents(docs)  # 切分并返回片段列表
