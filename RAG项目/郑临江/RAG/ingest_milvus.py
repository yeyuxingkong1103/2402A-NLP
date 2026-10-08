# -*- coding: utf-8 -*-
"""把本地文档解析、分块后灌入 Milvus（走 HybridRetriever.ingest）。

与离线库 ``OfflineRAG.add_file`` 走同一套「识别 → 解析 → 分块 → 向量化」链路，
区别是最后一步写入 Milvus（而非 kb.sqlite）；sparse 向量由服务端 BM25 函数自动生成。

运行前：
    1. 使用 rags_ 环境（含 pymilvus / sentence-transformers / jieba）：
           D:/an/envs/rags_/python.exe ingest_milvus.py ...
    2. Milvus 已启动：http://localhost:19530；
    3. bge-m3 权重存在：D:/modelscope/bge-m3（config.yaml 里的 embed_model）。

用法：
    python ingest_milvus.py <文件或目录> [更多文件或目录 ...]

示例：
    python ingest_milvus.py data/农业知识.81130062_47.pdf
    python ingest_milvus.py data            # 递归扫描目录下所有支持的文件

注意：
    - 图片类文件需要 paddleocr（仅 ocr_ 环境），在 rags_ 环境会解析失败被跳过；
    - ingest 每次都会新增 chunk_id，重复灌同一文档会产生重复数据；
      如需覆盖旧数据，先手动清空集合再灌（见 README / Milvus 管理端）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag2 import HybridRetriever, detect, load_config
from rag2.pipeline import chunk_text

# 可直接按文本读取的文件类别（与 rag2.pipeline._TEXT_KINDS 保持一致）
_TEXT_KINDS = {"text", "markdown", "code", "json", "jsonl", "csv", "xml", "html"}


def _read_text(path: Path, encoding: str | None = None) -> str:
    data = path.read_bytes()
    encodings = ([encoding] if encoding else []) + ["utf-8", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def extract_text(path: Path, info, describer=None) -> str:
    """与 OfflineRAG._extract_text 相同的解析路由。

    pdf/office → MinerU（失败回退 pdfplumber）；image → OCR（可选再叠加 VLM 描述）；文本类 → 直读。
    """
    if info.kind == "pdf":
        try:
            from rag2.mineru_parser import parse_pdf

            return parse_pdf(path).markdown
        except Exception:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return "\n\n".join((p.extract_text() or "") for p in pdf.pages)
    if info.kind == "image":
        from rag2.ocr import ocr_image

        text = ocr_image(path, save_txt=False).text
        if describer is not None:
            try:
                desc = (describer.describe(path) or "").strip()
            except Exception as exc:  # noqa: BLE001 - VLM 失败不阻断入库
                print(f"[警告] {path.name} VLM 描述失败：{exc}")
                desc = ""
            if desc:
                text = f"{text.strip()}\n\n【图片内容描述】\n{desc}".strip()
        return text
    if info.kind in _TEXT_KINDS:
        return _read_text(path, info.encoding)
    if info.kind == "office":
        try:
            from rag2.mineru_parser import parse_pdf

            return parse_pdf(path).markdown
        except Exception:
            return ""
    return ""


def collect_files(args: list[str]) -> list[Path]:
    """把参数展开成文件列表：目录递归扫描，文件原样保留。"""
    files: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file())
        else:
            files.append(p)
    return files


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    cfg = load_config()
    mc = cfg.milvus

    retriever = HybridRetriever(
        uri=mc.uri,
        collection=mc.collection,
        dim=mc.dim,
        embed_model=mc.embed_model,
        device=mc.device,
        analyzer=mc.analyzer,
        sparse_field=mc.sparse_field,
        rerank_model=mc.rerank_model,
    )

    # 图片理解：config.vlm.enabled 时构造一个可复用的 VLM 描述器（避免每张图重复建客户端）
    describer = None
    if cfg.vlm.enabled:
        from rag2.vlm import ImageDescriber

        describer = ImageDescriber.from_config(cfg.vlm)

    total = 0
    for f in collect_files(argv):
        try:
            info = detect(f)
        except FileNotFoundError:
            print(f"[不存在] {f}")
            continue

        text = extract_text(f, info, describer)
        if not text.strip():
            print(f"[跳过] {f.name}（{info.kind}：未提取到文本）")
            continue

        chunks = chunk_text(text)
        # page 固定填 0：当前解析只拿回纯文本、没保留页码（与离线库 add_file 行为一致）。
        n = retriever.ingest(
            chunks,
            metadatas=[{"doc_id": info.name, "source": info.name, "page": 0} for _ in chunks],
        )
        total += n
        print(f"[入库] {f.name}（{info.kind}）→ {n} 块")

    print(f"本次写入 {total} 块；集合「{mc.collection}」当前共 {retriever.count()} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
