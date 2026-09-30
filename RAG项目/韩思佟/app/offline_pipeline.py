"""离线建库：把医学 PDF 变成 Milvus 中可以检索的知识。

只记这一条主线即可：
PDF解析 → 清洗 → 分块 → BGE向量化 → Milvus同步。

“离线”表示只在新增或修改资料时运行，不会在每次聊天时运行。
为了让答辩代码容易读，本文件展示流程；复杂兼容和数据库安全检查放在
``app/internal/offline_engine.py``，这里调用它们，功能并没有删除。
"""
from __future__ import annotations

import argparse  # 读取终端中的命令和参数，例如 prepare --chunk-size 250。
import json  # 把检查结果变成 JSON 文本打印出来。
import re  # 用正则表达式删除空白，判断某一页是不是扫描页。
from pathlib import Path  # 用同一种写法兼容 Windows 路径和 Ubuntu 路径。

try:
    # 用 python -m app.offline_pipeline 启动时，从 app 包中导入内部实现。
    from app.internal import offline_engine as backend
except ModuleNotFoundError:
    # 在 IDE 中直接运行本文件时，改用相对当前目录的导入方式。
    from internal import offline_engine as backend


# ---------- 0. 公共配置：离线和在线必须使用同一个知识库 ----------

# __file__ 是当前文件；parents[1] 是项目根目录 D:/rag-roleplay。
BASE = Path(__file__).resolve().parents[1]
# Milvus collection 类似 MySQL 的表，在线程序也会读取这个名字。
COLLECTION = backend.COLLECTION
# 每块目标约 250 字；太大不精准，太小容易丢失上下文。
CHUNK_SIZE = backend.CHUNK_SIZE
# 太短的尾块会合并进上一块，避免产生没有意义的小片段。
MIN_CHUNK_SIZE = backend.MIN_CHUNK_SIZE

# 下列下划线名字来自internal：解析工具会真实运行；快照工具只供测试验证同步安全，答辩可跳过。
_sha256 = backend._sha256
_embedding_manifest = backend._embedding_manifest
_load_snapshot = backend._load_snapshot
_pymupdf_text = backend._pymupdf_text
_paddleocr_text = backend._paddleocr_text
_mineru_text = backend._mineru_text
_table_text = backend._table_text


def load_env(project_root: Path = BASE) -> None:
    """读取 ``.env.local``；project_root 是项目根目录。"""
    # 里面保存模型名和 Milvus 地址等配置，避免把配置写死在代码中。
    backend.load_env(project_root)


# ---------- 1. PDF 解析：把不同类型的 PDF 变成普通文字 ----------

def parse_document(
    pdf_path: Path,
    engine: str = "auto",
    complex_layout: bool = False,
    min_chars_per_page: int = 80,
    include_tables: bool = False,
) -> tuple[str, str]:
    """返回 ``(解析出的正文, 实际使用的解析器)``。

    pdf_path：PDF 路径；engine：auto/PyMuPDF/OCR/MinerU；
    complex_layout：复杂排版是否交给 MinerU；min_chars_per_page：少于多少字
    算扫描页；include_tables：是否再用 PDFPlumber 抽取表格。
    """
    pdf_path = Path(pdf_path)  # 即使调用者传来字符串，也统一转成路径对象。
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"不是有效PDF文件：{pdf_path}")
    allowed = {"auto", "pymupdf", "paddleocr", "mineru"}
    if engine not in allowed:  # 提前拒绝拼错的解析器名，错误更容易定位。
        raise ValueError(f"不支持的解析器：{engine}")
    if min_chars_per_page < 0:
        raise ValueError("min_chars_per_page不能小于0")

    # auto 表示让程序选择；复杂版式直接选 MinerU，其余先尝试快速的 PyMuPDF。
    used_engine = "mineru" if engine == "auto" and complex_layout else engine
    if used_engine in {"auto", "pymupdf"}:
        text, _density = _pymupdf_text(pdf_path)  # _density是返回但本层不用的密度统计，前导下划线表示“暂不使用”。
        used_engine = "pymupdf"
        if engine == "auto":
            pages = text.split("\n\f\n")  # \f是换页符；按页拆开后才能只OCR扫描页。
            sparse_pages = []  # 保存需要OCR的页码；enumerate同时给出页码和正文。
            for page_number, page in enumerate(pages):  # enumerate同时给正文和从0开始的页下标。
                visible_text = re.sub(r"\s+", "", page)  # 去空白后再统计真实字符数。
                if len(visible_text) < min_chars_per_page:
                    sparse_pages.append(page_number)
            if sparse_pages:
                # OCR 比文字层读取慢，所以只识别文字过少的页。
                ocr_text = _paddleocr_text(pdf_path, page_numbers=sparse_pages)
                ocr_pages = ocr_text.split("\n\f\n")
                if len(ocr_pages) != len(sparse_pages):
                    raise RuntimeError("PaddleOCR返回页数与待识别页数不一致")
                for page_number, page_text in zip(sparse_pages, ocr_pages):  # zip把原页下标与OCR结果一一配对。
                    pages[page_number] = page_text.strip() or pages[page_number]
                text = "\n\f\n".join(pages)  # 把处理后的页面重新合成全文。
                all_pages_are_scans = len(sparse_pages) == len(pages)
                used_engine = "paddleocr" if all_pages_are_scans else "pymupdf+paddleocr"
    elif used_engine == "paddleocr":
        text = _paddleocr_text(pdf_path)  # 用户明确指定时，整份 PDF 都走 OCR。
    else:
        text = _mineru_text(pdf_path)  # 标题、图片和公式较复杂时使用 MinerU。

    if include_tables:
        table_text = _table_text(pdf_path)  # PDFPlumber补充普通解析容易丢的表格。
        text = f"{text}\n\n{table_text}" if table_text else text
    return text.strip(), used_engine  # 同时返回引擎名，方便记录和答辩展示。


# ---------- 2. 清洗和分块：把全文变成大小合适的知识片段 ----------

def clean_text(raw_text: str) -> str:
    """输入PDF原文，删除空行、纯页码、目录点和重复空格，再返回正文。
    输入不是字符串或清洗后为空时停止，避免把坏数据送去分块。"""
    return backend.clean_text(raw_text)


def split_chunks(
    text: str,
    size: int = CHUNK_SIZE,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """输入正文，优先按段落和句末切到size附近，返回知识块列表。
    min_size是最短块；过短尾块会并入上一块，参数无效会报错。"""
    return backend.split_chunks(text, size=size, min_size=min_size)


def prepare_documents(
    project_root: Path = BASE,
    input_path: Path | None = None,
    engine: str = "auto",
    complex_layout: bool = False,
    include_tables: bool = False,
    size: int = CHUNK_SIZE,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[dict]:
    """执行“解析→清洗→分块”，保存中间结果并返回每份 PDF 的摘要。"""
    project_root = Path(project_root)
    load_env(project_root)  # 先加载 MinerU 命令等环境配置。
    raw_dir = project_root / "data" / "raw"
    # 传 input_path 就处理一个文件；不传就处理 data/raw 中所有 PDF。
    if input_path:
        pdf_files = [Path(input_path)]  # 指定文件时，列表里只有这一份PDF。
    else:
        pdf_files = sorted(raw_dir.glob("*.pdf"))  # 否则找出data/raw中的全部PDF。
    if not pdf_files:
        raise RuntimeError("data/raw中没有PDF文件")

    results = []  # 用列表收集每份 PDF 的处理结果，最后返回给调用者。
    for pdf_path in pdf_files:
        raw_text, used_engine = parse_document(
            pdf_path, engine, complex_layout, include_tables=include_tables
        )
        if not raw_text:
            raise RuntimeError(f"{pdf_path.name}没有解析出文字")
        cleaned_text = clean_text(raw_text)  # 先清洗，避免噪声进入知识库。
        chunks = split_chunks(cleaned_text, size=size, min_size=min_size)
        if not chunks:
            raise RuntimeError(f"{pdf_path.name}清洗后没有可用知识块")

        source = pdf_path.stem  # 文件名去掉 .pdf，作为知识来源标识。
        parsed_path = project_root / "data" / "parsed" / f"{source}.txt"
        backend._write_text(parsed_path, raw_text)  # 保存解析原文，方便人工抽查。
        backend._write_json(parsed_path.with_suffix(".meta.json"), {
            "source": pdf_path.name,
            "engine": used_engine,
            "characters": len(raw_text),
        })
        cleaned_path = project_root / "data" / "cleaned" / f"{source}.txt"
        backend._write_text(cleaned_path, cleaned_text)  # 保存清洗后的全文。
        records = []  # 每个字典代表一块知识；JSONL中会一行保存一个字典。
        for index, chunk in enumerate(chunks):
            record = {"id": f"{source}_{index:03d}", "text": chunk,
                      "source": source, "index": index}
            records.append(record)
        chunks_path = project_root / "data" / "chunks" / f"{source}.jsonl"
        backend._write_jsonl(chunks_path, records)  # 一行一个块，供下一步向量化。
        summary = {"source": pdf_path.name, "engine": used_engine,
                   "characters": len(raw_text), "chunks": len(chunks)}
        results.append(summary)
        print(f"PREPARED: {pdf_path.name} 引擎={used_engine} "
              f"字符={len(raw_text)} 块={len(chunks)}")
    return results


# ---------- 3. BGE向量化：把文字转换成计算机可比较的数字 ----------

def embed_chunks(
    project_root: Path = BASE,
    model_name: str | None = None,
    batch_size: int = 16,
    chunks_dir: Path | None = None,
    output_dir: Path | None = None,
    model_factory=None,
) -> list[dict]:
    """读取知识块，用本地BGE生成向量，返回各文件摘要。
    model_name选模型；batch_size是一批块数；两个dir可改输入输出目录；
    model_factory只供单元测试，答辩跳过。模型或数据缺失时停止，不写假向量。
    产物是data/embeddings中的JSONL；向量维度由所选BGE模型决定。"""
    # internal会归一化向量；这一阶段全部本地运行，不调用DeepSeek或其他在线API。
    return backend.embed_chunks(
        project_root, model_name, batch_size, chunks_dir, output_dir, model_factory
    )


# ---------- 4. Milvus同步：把文字、来源和向量写进向量数据库 ----------

def sync_milvus(
    project_root: Path = BASE,
    embeddings_dir: Path | None = None,
    report_path: Path | None = None,
    *,
    prune: bool = False,
    dry_run: bool = False,
    reset: bool = False,
    client_factory=None,
    data_type=None,
    snapshot_manifest: Path | None = None,
) -> dict:
    """把向量文件同步到Milvus，返回新增、更新、保留和删除数量。
    dry_run只预览；prune按snapshot_manifest清单删旧块；reset重建集合；
    client_factory和data_type只供测试，答辩跳过。字段、维度或哈希错误时停止。
    返回changes字典：added新增、updated更新、stale_retained保留、pruned删除。"""
    # internal 还会检查字段、向量维度、文件哈希并验证写入，防止误删知识。
    return backend.sync_milvus(
        project_root, embeddings_dir, report_path, prune=prune, dry_run=dry_run,
        reset=reset, client_factory=client_factory, data_type=data_type,
        snapshot_manifest=snapshot_manifest,
    )


def _parser_status() -> dict[str, bool]:
    """检查 PyMuPDF、PDFPlumber、PaddleOCR 和 MinerU 是否可用。"""
    return backend._parser_status()


# ---------- 5. 命令行入口：把用户命令分发给上面四个阶段 ----------

def build_parser() -> argparse.ArgumentParser:
    """定义 check、prepare、embed、sync 四个命令及其参数。"""
    parser = argparse.ArgumentParser(description="医学RAG离线知识库流水线")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="检查PDF解析器")
    check.add_argument("--require-all", action="store_true", help="缺少任一解析器就失败")
    prepare = commands.add_parser("prepare", help="PDF解析、清洗并分块")
    prepare.add_argument("--input", type=Path, help="单个PDF；省略则处理data/raw")
    prepare.add_argument("--engine", choices=["auto", "pymupdf", "paddleocr", "mineru"],
                         default="auto", help="PDF解析器；auto会自动补OCR")
    prepare.add_argument("--complex-layout", action="store_true", help="复杂排版使用MinerU")
    prepare.add_argument("--tables", action="store_true", help="用PDFPlumber追加表格")
    prepare.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="目标块字数")
    prepare.add_argument("--min-size", type=int, default=MIN_CHUNK_SIZE, help="最短块字数")
    embed = commands.add_parser("embed", help="用本地BGE生成向量")
    embed.add_argument("--model", help="models目录中的模型文件夹名")
    embed.add_argument("--batch-size", type=int, default=16, help="每批向量化条数")
    sync = commands.add_parser("sync", help="安全更新Milvus知识库")
    sync.add_argument("--embeddings-dir", type=Path, help="向量JSONL目录")
    sync.add_argument("--report", type=Path, help="同步报告保存路径")
    sync.add_argument("--dry-run", action="store_true", help="只预览，不修改数据库")
    sync.add_argument("--prune", action="store_true", help="删除快照中不存在的旧块")
    sync.add_argument("--snapshot-manifest", type=Path, help="prune所需的完整快照清单")
    sync.add_argument("--reset", action="store_true", help="删除并重建collection")
    return parser


def main() -> None:
    """读取命令，并调用对应的离线阶段。"""
    parser = build_parser()
    args = parser.parse_args()  # 把终端文字转换成 args.command 等 Python 变量。
    load_env(BASE)
    # 三个正式步骤要按prepare→embed→sync依次运行；这里根据本次命令只执行一步。
    if args.command == "check":
        status = _parser_status()
        print(json.dumps(status, ensure_ascii=False))
        if args.require_all and not all(status.values()):
            missing = "、".join(name for name, ready in status.items() if not ready)
            parser.error(f"解析器未就绪：{missing}")
    elif args.command == "prepare":
        prepare_documents(BASE, args.input, args.engine, args.complex_layout, args.tables,
                          args.chunk_size, args.min_size)
    elif args.command == "embed":
        embed_chunks(BASE, args.model, args.batch_size)
    else:  # 只有四个合法命令，所以最后一种必然是 sync。
        report = sync_milvus(BASE, args.embeddings_dir, args.report, prune=args.prune,
                             dry_run=args.dry_run, reset=args.reset,
                             snapshot_manifest=args.snapshot_manifest)
        changes = report["changes"]  # 取出本次同步的新增、更新、保留和删除数量。
        print(f"MILVUS_{report['status'].upper()}: added={changes['added']} "
              f"updated={changes['updated']} retained={changes['stale_retained']} "
              f"pruned={changes['pruned']} restart_required=true")  # 重启在线服务后，BM25才会读取最新原文。


if __name__ == "__main__":
    main()  # 直接启动才执行；被单元测试 import 时不会自动运行。
