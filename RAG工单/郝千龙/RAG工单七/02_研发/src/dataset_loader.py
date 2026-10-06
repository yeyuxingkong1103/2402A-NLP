# -*- coding: utf-8 -*-
# 【语料加载模块 · dataset_loader.py】优先加载ccf_competition.zip，缺失时明确报错并自动回退招股书语料
# 工单编号：人工智能NLP-RAG-功能测试及评估

"""语料加载层（测试流水线入口）。

数据来源优先级：
1. ``data/ccf_competition.zip``：工单指定的 CCF 竞赛语料压缩包。
   支持压缩包内任意层级的 ``*.pdf``（含嵌套 zip），同时自动识别
   ``sample_questions.pdf`` 一类示例问题文件（默认跳过，不作为检索语料）。
2. 压缩包不存在时：打印明确的放置说明，并自动回退加载 ``data/`` 目录下
   直接存放的 PDF（本次评测使用真实存在的《招股说明书1.pdf》演示同一套
   测试与 RAGAS 评估流程）。
3. 两者都不存在时抛出 ``DatasetNotFoundError``，提示如何放置数据。

说明：本机当前不存在 ``ccf_competition.zip``，本模块不会编造其文件清单；
zip 到位后无需改代码，重新运行 run_evaluation.py 即可对真实竞赛语料评测。
"""
import io
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pymupdf

# 工单指定的竞赛语料压缩包名称
CCF_ZIP_NAME = "ccf_competition.zip"
# 压缩包中若带有示例问题 PDF，不作为被检索语料（按文件名关键字识别）
QUESTION_FILE_KEYWORDS = ("sample_question", "question", "问题")
# 支持的文档后缀
PDF_SUFFIX = ".pdf"


class DatasetNotFoundError(FileNotFoundError):
    """竞赛 zip 与本地回退语料均不存在时抛出。"""


@dataclass
class Document:
    """单篇 PDF 文档。

    :ivar file_name: 文档名（zip 内为相对路径，回退模式为文件名）
    :ivar pages: 逐页纯文本（页码 = 列表下标 + 1）
    :ivar source: 来源标记，ccf_zip / fallback_pdf
    """

    file_name: str
    pages: List[str]
    source: str = "ccf_zip"
    meta: Dict[str, str] = field(default_factory=dict)


def _read_pdf_pages(buffer: bytes) -> List[str]:
    """从字节流解析 PDF，返回逐页文本。

    :param buffer: PDF 文件字节
    :return: 每页纯文本组成的列表（空页保留空串）
    """
    pages: List[str] = []
    with pymupdf.open(stream=buffer, filetype="pdf") as pdf:
        for page in pdf:
            pages.append(page.get_text("text"))
    return pages


def _is_question_file(name: str) -> bool:
    """判断 zip 内文件是否为示例问题文件（不应进入检索语料）。

    :param name: zip 内条目名
    :return: 是示例问题文件返回 True
    """
    lower = name.lower()
    return lower.endswith(PDF_SUFFIX) and any(k in lower for k in QUESTION_FILE_KEYWORDS)


def _collect_pdfs_from_zip(zf: zipfile.ZipFile, prefix: str = "") -> List[Document]:
    """递归收集 zip（含嵌套 zip）内全部业务 PDF。

    :param zf: 已打开的 ZipFile 对象
    :param prefix: 嵌套层级路径前缀（仅用于文档名展示）
    :return: Document 列表
    """
    docs: List[Document] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        lower = info.filename.lower()
        if lower.endswith(PDF_SUFFIX):
            if _is_question_file(info.filename):
                # 示例问题文件跳过，避免污染检索库
                continue
            pages = _read_pdf_pages(zf.read(info))
            docs.append(Document(
                file_name=prefix + os.path.basename(info.filename),
                pages=pages, source="ccf_zip",
                meta={"zip_entry": info.filename}))
        elif lower.endswith(".zip"):
            # 兼容嵌套压缩包：递归展开
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as nested:
                docs.extend(_collect_pdfs_from_zip(
                    nested, prefix=prefix + os.path.basename(info.filename) + "!/"))
    return docs


def _load_ccf_zip(zip_path: str) -> List[Document]:
    """加载 ccf_competition.zip 中的全部 PDF。

    :param zip_path: 压缩包绝对路径
    :return: Document 列表；包内无 PDF 时抛出异常提示
    """
    with zipfile.ZipFile(zip_path) as zf:
        docs = _collect_pdfs_from_zip(zf)
    if not docs:
        raise DatasetNotFoundError(
            f"{zip_path} 中未找到任何业务 PDF（仅识别 *.pdf；示例问题文件会被跳过）。"
            "请检查压缩包结构是否与工单要求一致。")
    return docs


def _load_fallback_pdfs(data_dir: str) -> List[Document]:
    """回退模式：加载 data 目录下直接存放的 PDF。

    :param data_dir: 语料目录
    :return: Document 列表
    """
    docs: List[Document] = []
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(PDF_SUFFIX) and not name.startswith("~$"):
            full = os.path.join(data_dir, name)
            with pymupdf.open(full) as pdf:
                pages = [page.get_text("text") for page in pdf]
            docs.append(Document(
                file_name=name, pages=pages, source="fallback_pdf",
                meta={"path": full}))
    return docs


def _print_zip_missing_help(data_dir: str, zip_path: str) -> None:
    """zip 缺失时打印明确的放置说明（控制台与日志均可看到）。"""
    print("=" * 72)
    print("[语料提示] 未检测到工单指定语料 ccf_competition.zip")
    print(f"  期望位置：{zip_path}")
    print("  放置方法：将竞赛语料压缩包原样放入上述目录后重新运行，无需改代码。")
    print("  本次处理：自动回退到 data/ 目录下的招股书 PDF 语料，")
    print("            用同一套检索测试流水线与 RAGAS 四项指标完成功能测试及评估演示。")
    print("=" * 72)


def load_corpus(data_dir: str,
                ccf_zip_name: Optional[str] = None) -> Dict[str, object]:
    """加载评测语料（竞赛 zip 优先，缺失自动回退）。

    :param data_dir: 语料目录（通常为项目根下的 data/）
    :param ccf_zip_name: 可选，自定义竞赛 zip 文件名
    :return: ``{"documents": Document列表, "source": 来源, "data_dir": 目录,
             "zip_path": zip路径或空串}``
    :raises DatasetNotFoundError: zip 与回退 PDF 均不存在
    """
    zip_name = ccf_zip_name or CCF_ZIP_NAME
    zip_path = os.path.join(data_dir, zip_name)

    if os.path.exists(zip_path):
        documents = _load_ccf_zip(zip_path)
        source = "ccf_zip"
        print(f"[语料加载] 已加载竞赛语料：{zip_path}，共 {len(documents)} 篇 PDF。")
    else:
        _print_zip_missing_help(data_dir, zip_path)
        documents = _load_fallback_pdfs(data_dir)
        if not documents:
            raise DatasetNotFoundError(
                "既未找到 ccf_competition.zip，data/ 目录下也没有任何 PDF 语料。\n"
                f"请将 ccf_competition.zip 放入目录：{data_dir}\n"
                "或放入至少一份招股书/研报 PDF 作为回退语料后重试。")
        source = "fallback_pdf"
        print(f"[语料加载] 回退语料 {len(documents)} 篇："
              + "、".join(d.file_name for d in documents))

    total_pages = sum(len(d.pages) for d in documents)
    return {
        "documents": documents,
        "source": source,
        "data_dir": data_dir,
        "zip_path": zip_path if source == "ccf_zip" else "",
        "doc_count": len(documents),
        "page_count": total_pages,
    }
