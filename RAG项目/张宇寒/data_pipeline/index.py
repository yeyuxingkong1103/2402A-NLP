"""离线数据处理和写入向量数据库的总编排文件。

本文件把已经拆开的能力按正确顺序串起来：

``读取文件 -> 解析法律结构 -> 清洗校验 -> 分块 -> 生成向量 -> 写入 Milvus``

它既支持公共知识库，也保留了用户私有资料的通用处理入口。真正的读取、解析、
分块和向量化算法分别位于 load.py、parse.py、chunk.py 和 embedding.py；这里主要
负责决定先调用谁、后调用谁，以及写库前怎样保护已有数据。
"""

# 让类型注解延迟解析，避免某些类型只用于说明时也必须在运行期立即加载。
from __future__ import annotations

# json 用于把处理结果转换为可读的 JSON 文件。
import json
# Path 用于安全地拼接和检查 Windows/Linux 文件路径。
from pathlib import Path
# Any 表示这里可以接收真实 MilvusStore，也可以在测试中接收模拟对象。
from typing import Any

# Settings 是配置对象类型；get_settings 在未传配置时读取项目配置。
from backend.app.config import Settings, get_settings
# ModelGateway 统一封装嵌入模型调用。
from backend.app.models import ModelGateway
# MilvusStore 封装公共集合和用户私有集合的写入操作。
from backend.app.storage.vector import MilvusStore

# chunk_document 按文档结构和长度把长文本切成可以检索的小块。
from .chunk import chunk_document
# RECORD_FIELDS 定义各集合的主键和正文；另外两个函数负责清洗与质量校验。
from .clean import RECORD_FIELDS, clean_public_collections, validate_public_records
# embed 处理通用分块；另两个函数负责公共数据的分批与向量化。
from .embedding import embed, embed_public_batch, make_public_batches
# extract 读取纯文本；extract_content 返回带页面信息的结果；complete_text 检查读取是否完整。
from .load import complete_text, extract, extract_content
# parse 模块负责把普通文本整理成条文、案例、证据等结构化记录。
from .parse import (
    # SOURCE_FILES 保存每类公共集合对应的原始文件名。
    SOURCE_FILES,
    # build_citations 根据记录之间的关系建立引用记录。
    build_citations,
    # build_elements 从条文、解释和案例中构建民事要素记录。
    build_elements,
    # parse_document 根据资料类型选择相应解析规则。
    parse_document,
)


# 所有公共集合名称；最后的 civil_citations 是由程序生成的引用集合。
PUBLIC_COLLECTIONS = (*SOURCE_FILES.keys(), "civil_citations")

# 每个集合允许写入 Milvus 的业务字段白名单。
# 使用白名单能防止临时字段、调试字段或超大对象被意外存进数据库。
PUBLIC_PAYLOAD_FIELDS = {
    "civil_code_articles": {"id", "article_number", "law_name", "chapter", "article_content"},
    "civil_interpretations": {"id", "law_name", "part_name", "document_number", "content"},
    "civil_cases": {"case_id", "title", "case_number", "cause_of_action", "summary"},
    "civil_elements": {"serial_number", "title", "case_number", "case_summary"},
    "civil_evidence": {"evidence_id", "source_case_id", "title", "rule_name", "content"},
    "civil_processes": {"process_id", "title", "content"},
    "civil_questions": {"question_id", "question", "content"},
    "civil_citations": {"citation_id", "source_collection", "source_id", "target_collection", "target_id", "article_number", "relation_type", "content"},
}
# 从 RECORD_FIELDS 中取出每个集合的正文字段名称。
# fields[0] 是主键字段，fields[1] 是正文；这里只需要正文。
PUBLIC_TEXT_FIELDS = {}
for collection, fields in RECORD_FIELDS.items():
    PUBLIC_TEXT_FIELDS[collection] = fields[1]


def build_public_records(data_dir: Path, output_dir: Path) -> dict:
    """处理完整公共数据集并保存 JSON，但不连接 Milvus。

    ``data_dir`` 是原始资料目录，``output_dir`` 是处理结果目录。
    返回字典中包含全部集合、质量报告和已保存文件的路径，供 main.py 决定
    是仅供预览，还是继续调用追加/重建函数入库。
    """

    # 读取 SOURCE_FILES 中列出的原始文件，并解析、清洗为多个集合。
    collections = _read_public_sources(Path(data_dir))
    # civil_elements 不是直接读取的文件，而是由条文、司法解释和案例共同生成。
    collections["civil_elements"] = build_elements(collections["civil_code_articles"], collections["civil_interpretations"], collections["civil_cases"])
    # 派生新集合后再整体清洗一次，保证所有记录格式一致。
    collections, _ = clean_public_collections(collections)
    # 根据各集合中的法条编号和来源关系，生成可以追溯的引用集合。
    collections["civil_citations"] = build_citations(collections)
    # 检查必填字段、重复主键等问题，并生成质量统计。
    quality = validate_public_records(collections)
    # 将每个集合分别保存为 JSON，方便人工检查且不用每次重新解析原文件。
    paths = save_public_records(collections, output_dir)
    # 质量报告也保存成 JSON；返回的 report_path 是实际文件路径。
    report_path = save_processed_result(quality, Path(output_dir) / "quality_report.json")
    # 统一返回本次处理产生的全部信息，供命令行入口继续使用。
    return {"collections": collections, "quality": quality, "paths": paths, "quality_report": report_path}


# 旧代码可能仍导入 build_public；这个别名让旧调用继续指向新函数。
build_public = build_public_records


def process_public_file(path: Path, *, collection: str, output_dir: Path, pdf_method: str = "auto") -> dict:
    """处理一份公共资料以便预览，不生成向量，也不连接数据库。

    ``path`` 是文件地址，``collection`` 指明资料类型，``pdf_method`` 决定 PDF
    使用本地文本、OCR 或多模态读取。返回解析后的集合、质量结果和保存路径。
    """

    # collection 必须是可直接解析的公共集合，自动派生的 civil_elements 不能手工导入。
    if collection not in SOURCE_FILES or collection == "civil_elements":
        raise ValueError("请选择条文、司法解释、案例、证据、流程或问答集合。")
    # 读取文件并保留每一页的文本、读取方式和失败信息。
    result = extract_content(Path(path), pdf_method=pdf_method)
    # 如果存在失败页或读取结果不完整，就直接报错，不保存残缺法律资料。
    complete_text(result)
    # 拼接各页原文；不用 result.text 是为了避免把读取器附加的页码标签混进正文。
    text = "\n".join(page["text"] for page in result.pages)
    # 按指定集合的法律资料规则，把普通文本解析成结构化记录。
    parsed = parse_document(text, collection=collection)
    # 清洗这一个集合；下划线接收暂时不需要使用的第二个返回值。
    collections, _ = clean_public_collections({collection: parsed["records"]})
    # 没有任何有效记录通常表示用户选错分类，或者文件格式不符合解析规则。
    if not collections[collection]:
        raise ValueError("没有解析出有效法律记录，请检查分类和资料格式。")
    # 在保存之前检查主键、必填内容和重复数据。
    quality = validate_public_records(collections)
    # 把结构化记录保存到单独的预览目录。
    paths = save_public_records(collections, output_dir)
    # 同时保存质量报告，便于人工判断这份资料能否加入知识库。
    save_processed_result(quality, Path(output_dir) / "quality_report.json")
    # 返回后，main.py 会显示记录数量和保存位置。
    return {"collections": collections, "quality": quality, "paths": paths}


def _vector_store(vector_store: Any | None = None, settings: Settings | None = None):
    """取得向量存储对象；测试传入模拟对象，正式运行创建 MilvusStore。"""

    # 优先使用调用者传入的对象；没有传入时才读取配置并连接 Milvus。
    return vector_store if vector_store is not None else MilvusStore(settings or get_settings())


def index(rows: list[dict], *, scope: str, collection: str | None = None, user_id: str = "", document_id: str = "", session_id: str = "", vector_store: Any | None = None, settings: Settings | None = None) -> list[dict]:
    """把已经带向量的分块写入公共集合或用户私有集合。

    ``rows`` 是待写入记录；``scope`` 为 ``private`` 时按用户、文件和会话隔离，
    其他情况按 ``collection`` 写入公共库。函数返回最终写入的数据，便于上层记录
    本次处理结果。
    """

    # 空列表不需要连接数据库，直接返回可以节省一次无意义连接。
    if not rows:
        return []
    # 获取真实或测试用的向量存储对象。
    store = _vector_store(vector_store, settings)
    # private 表示用户上传资料，必须补上用户、文件、会话等隔离字段。
    if scope == "private":
        # payload 用于收集格式统一后的全部私有分块。
        payload = []
        # enumerate 同时提供分块顺序 position 和原始记录 source。
        for position, source in enumerate(rows):
            # 复制字典，避免给原始 rows 增加字段而产生意外副作用。
            row = dict(source)
            # 一次补齐私有向量集合要求的字段。
            row.update(
                # 优先使用已有 chunk_id；没有时按文件名和顺序生成稳定编号。
                chunk_id=str(row.get("chunk_id") or f"{document_id or 'document'}_chunk_{position:03d}"),
                # user_id 保证查询时只检索当前用户的资料。
                user_id=user_id,
                # document_id 表示这个分块来自哪一份上传文件。
                document_id=document_id,
                # session_id 限制资料只在对应咨询会话中使用。
                session_id=session_id,
                # chunk_index 保存分块原本的先后顺序，方便恢复上下文。
                chunk_index=int(row.get("chunk_index", position)),
                # 统一正文名称为 content，并兼容不同处理阶段使用的旧字段名。
                content=str(row.get("content") or row.get("text") or row.get("embedding_text") or ""),
                # 文件名用于前端展示证据来源。
                file_name=str(row.get("file_name") or ""),
            )
            # 把整理后的这一条记录加入待写入列表。
            payload.append(row)
        # 一次写入当前用户的私有资料集合。
        store.insert_private(payload)
        # 返回实际写入的结构化记录。
        return payload

    # 公共数据必须明确集合，否则无法确定主键、正文和存储位置。
    if not collection:
        raise ValueError("public 入库必须提供 collection")
    # 收集格式统一后的公共记录。
    payload = []
    # 逐条复制并补充公共检索所需字段。
    for source in rows:
        row = dict(source)
        # 原记录没有 collection 时，写入当前目标集合名称。
        row.setdefault("collection", collection)
        # 统一正文为 content，同时兼容 text 和 embedding_text。
        row.setdefault("content", str(row.get("content") or row.get("text") or row.get("embedding_text") or ""))
        payload.append(row)
    # 调用存储层将记录写进指定的公共 Milvus 集合。
    store.insert_public(collection, payload)
    # 返回最终写入记录，便于上层统计或测试。
    return payload


def file_metadata(path: Path) -> dict:
    """读取文件名、扩展名和字节大小，不读取文件正文。"""

    # 把字符串或 Path 统一转换成 Path 对象。
    source = Path(path)
    # stat() 从文件系统读取大小等基础属性。
    stat = source.stat()
    # suffix.lower() 将 .PDF 和 .pdf 统一为 .pdf，方便后续判断格式。
    return {"file_name": source.name, "suffix": source.suffix.lower(), "size_bytes": stat.st_size}


def save_processed_result(result: dict | list[dict], output_path: Path) -> Path:
    """把字典或记录列表保存为 UTF-8 JSON，并返回保存路径。"""

    # 统一目标路径的类型。
    target = Path(output_path)
    # parents=True 会补齐多层目录；exist_ok=True 表示目录已存在也不报错。
    target.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保留中文，indent=2 让 JSON 便于人工阅读。
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 返回 Path，调用者可以显示或继续记录该文件位置。
    return target


# 下面两个别名兼容旧代码中的简短函数名。
save = save_processed_result
metadata = file_metadata


def process(path: Path, *, scope: str, collection: str | None = None, document_type: str | None = None, user_id: str = "", document_id: str = "", session_id: str = "", chunk_size: int = 1000, model=None, vector_store=None) -> dict:
    """执行一份文档的完整通用管道：读取、解析、分块、向量化和可选入库。

    这个函数主要给用户上传资料路线复用。传入自定义 ``model`` 而不传
    ``vector_store`` 时只生成结果、不连接数据库，方便测试和预览。
    """

    # 统一文件路径对象。
    source = Path(path)
    # 读取嵌入批次大小、预期维度等配置。
    settings = get_settings()
    # 记录模型是否由调用者传入；测试时通常会传一个假的模型。
    custom_model = model is not None
    # 有自定义模型就直接用，否则创建项目统一模型网关。
    gateway = model or ModelGateway(settings)
    # 显式传入存储对象就使用它；仅自定义模型时不自动连库；正式运行则创建 MilvusStore。
    store = vector_store if vector_store is not None else (None if custom_model else MilvusStore(settings))
    # 从文件读取可解析的纯文本。
    text = extract(source)
    # 根据 collection/document_type 将文本解析为法律结构。
    parsed = parse_document(text, document_type=document_type, collection=collection, source_path=source)
    # details 是之后每个分块都会携带的来源和隔离信息。
    details = {
        # 展开基础文件信息：文件名、后缀和大小。
        **file_metadata(source),
        # 优先使用数据库给出的 document_id，否则用文件名主体作为临时编号。
        "file_id": document_id or source.stem,
        # 用户编号用于私有资料隔离。
        "user_id": user_id,
        # 会话编号限制这份证据只服务于当前咨询。
        "session_id": session_id,
        # scope 说明这是公共资料还是用户私有资料。
        "scope": scope,
        # 公共资料使用 collection 区分类别；没有时保存空字符串。
        "collection": collection or "",
    }
    # 按法律结构和 chunk_size 将长文拆成较小的检索单元。
    chunks = chunk_document(parsed, details, chunk_size=chunk_size)
    # 调用嵌入模型给每个分块生成向量，并检查批次大小和正式模型的向量维度。
    embedded = embed(chunks, gateway, batch_size=settings.embedding_batch_size, max_characters=settings.embedding_batch_max_chars, expected_dimension=None if custom_model else settings.embedding_dim)
    # 有存储对象就写入 Milvus；预览/测试模式则把带向量记录直接当作结果。
    indexed = index(embedded, scope=scope, collection=collection, user_id=user_id, document_id=document_id or source.stem, session_id=session_id, vector_store=store) if store is not None else embedded
    # 返回每个处理中间结果，方便上层展示、记录状态和定位问题。
    return {
        # 文件及隔离信息。
        "metadata": details,
        # 原始读取文本和读取方式。
        "extracted": {"text": text, "extraction_method": "text"},
        # 法律结构化解析结果。
        "parsed": parsed,
        # 尚未向量化的分块。
        "chunks": chunks,
        # 已带 embedding 的分块。
        "embedded_chunks": embedded,
        # 最终写入或准备写入的数据。
        "indexed_chunks": indexed,
    }


def process_user_file(path: Path) -> dict:
    """快速预览用户文件的基础信息和文本分块，不生成向量、不入库。"""

    # 将输入路径统一为 Path。
    source = Path(path)
    # 读取文件名、后缀和大小。
    details = file_metadata(source)
    # 内层先读取，再解析，最后把解析结果连同文件信息交给分块器。
    chunks = chunk_document(parse_document(extract(source)), details)
    # 预览接口只需要每块文字，因此从分块字典中取出 text。
    return {"metadata": details, "chunks": [row.get("text", "") for row in chunks]}


def process_user_document(path, file_id: str | None = None, user_id: str = "", session_id: str | None = None, document_type: str | None = None, chunk_size: int = 1000, model=None, vector_store=None):
    """用户上传资料的完整处理入口，固定使用 private 私有范围。"""

    # 复用 process，统一完成读取、解析、分块、向量化和私有入库。
    # ``or ""`` 把可选的 None 转成下游存储层更容易处理的空字符串。
    return process(path, scope="private", user_id=user_id, document_id=file_id or "", session_id=session_id or "", document_type=document_type, chunk_size=chunk_size, model=model, vector_store=vector_store)


def ocr_document(path: str) -> dict:
    """兼容旧调用：读取一份文档并只返回识别后的完整文本。"""

    # extract_content 会按文件类型选择文本提取、OCR 或配置的读取方案。
    return {"text": extract_content(Path(path)).text}


def _source(data_dir: Path, collection: str) -> Path:
    """根据集合名称找到它应读取的公共原始文件，并确认文件存在。"""

    # SOURCE_FILES[collection] 给出该集合约定的原始文件名。
    path = Path(data_dir) / SOURCE_FILES[collection]
    # 缺少任何一个公共来源都立即停止，避免构建出表面成功但不完整的知识库。
    if not path.is_file():
        raise FileNotFoundError(f"{collection} 缺少原始文件：{path}")
    # 把已确认存在的路径交给读取器。
    return path


def _read_public_sources(data_dir: Path) -> dict[str, list[dict]]:
    """读取并解析 SOURCE_FILES 中约定的全部公共原始资料。"""

    # collections 的键是 Milvus 集合名，值是这个集合的结构化记录列表。
    collections = {}
    # 遍历每一种公共资料，保证完整建库时不会漏掉约定集合。
    for collection in SOURCE_FILES:
        # 找到并检查该集合的原始文件。
        path = _source(data_dir, collection)
        # 依次读取文本、按集合规则解析，并取出 records 记录列表。
        collections[collection] = parse_document(extract(path), collection=collection, source_path=path)["records"]
    # 整体清洗，统一字符串、空值和记录格式；第二个返回值在此不使用。
    cleaned, _ = clean_public_collections(collections)
    # 返回可继续生成派生集合的数据。
    return cleaned


def save_public_records(collections: dict[str, list[dict]], output_dir: Path) -> dict[str, Path]:
    """把每个公共集合分别保存成一个 JSON 文件。"""

    # 统一输出目录路径。
    target = Path(output_dir)
    # 输出目录不存在时自动创建。
    target.mkdir(parents=True, exist_ok=True)
    # paths 保存“集合名 -> 实际 JSON 路径”，供上层展示和记录。
    paths = {}
    # 每个集合独立保存，方便单独检查和重新利用处理结果。
    for collection, rows in collections.items():
        # 文件名直接使用集合名，例如 civil_cases.json。
        paths[collection] = save_processed_result(rows, target / f"{collection}.json")
    # 返回所有保存路径。
    return paths


def _check_public_lengths(collection: str, rows: list[dict]) -> None:
    """检查字段类型和 UTF-8 字节长度，避免向量化后才发现无法入库。"""

    # 读取当前集合约定的主键字段和正文字段。
    primary_field, content_field = RECORD_FIELDS[collection]
    # 逐条检查，错误信息可以精确指出集合和字段。
    for row in rows:
        # 主键和正文必须都是字符串，Milvus schema 才能稳定接收。
        for key in (primary_field, content_field):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{collection}.{key} 必须是字符串，不能直接保存数字、列表或对象。")
        # 检查这个集合所有允许入库的业务字段。
        for key in PUBLIC_PAYLOAD_FIELDS[collection]:
            # 取出字段值；缺失字段会得到 None。
            value = row.get(key)
            # 非字符串的可选字段不在这里做长度检查，直接进入下一字段。
            if not isinstance(value, str):
                continue
            # 主键最多 128 字节，正文最多 50,000 字节，其他说明字段最多 2,048 字节。
            limit = 128 if key == primary_field else (50000 if key == PUBLIC_TEXT_FIELDS[collection] else 2048)
            # Milvus 的 VARCHAR 限制按字节计算，中文 UTF-8 通常不止一个字节。
            if len(value.encode("utf-8")) > limit:
                raise ValueError(f"{collection}.{key} 长度超过 {limit} 字节，请整理原记录；不会截短入库。")


def validate_public_input(collections: dict[str, list[dict]]) -> None:
    """检查整批数据；这里只检查，不清洗、不改写法律原文。"""

    # 先执行数据层面的完整性、必填字段和重复主键检查。
    validate_public_records(collections)
    # 再逐集合检查 Milvus 字段类型和长度限制。
    for collection, rows in collections.items():
        _check_public_lengths(collection, rows)


def _public_rows(rows: list[dict], model, dimension: int | None, collection: str = "") -> list[dict]:
    """先转向量，再只保留这个集合允许保存的字段。"""

    # 模型调用前先检查，避免为最终无法保存的数据支付嵌入费用。
    _check_public_lengths(collection, rows)
    # 正式入库必须传模型；保留无模型分支是为了结构化测试。
    if model:
        # embedded_rows 收集各批次向量化后的记录。
        embedded_rows = []
        # 同时限制每批记录数和字符数，避免模型接口请求过大。
        for batch in make_public_batches(collection, rows, max_records=8, max_characters=30000):
            # 向量化当前批次并检查实际维度，再加入总结果。
            embedded_rows.extend(embed_public_batch(collection, batch, model, expected_dimension=dimension))
        # 后面的字段筛选应针对已经带 embedding 的记录。
        rows = embedded_rows
    # output 是最终可以交给 Milvus insert 的数据列表。
    output = []
    # 逐条生成安全的入库 payload。
    for row in rows:
        # 每条记录都从空字典开始，只复制白名单字段。
        payload = {}
        for key, value in row.items():
            # 不在白名单中的解析辅助字段不会写入数据库。
            if key in PUBLIC_PAYLOAD_FIELDS[collection]:
                payload[key] = value
        # 有模型时复制已校验向量；无模型的测试分支使用空向量占位。
        payload["embedding"] = row["embedding"] if model else []
        # 保存这一条整理好的 Milvus 记录。
        output.append(payload)
    # 返回全部待写入记录。
    return output


def _ensure_public_collection(client, collection: str, dimension: int | None) -> None:
    """确认 Milvus 集合结构正确；集合不存在时按项目约定创建。"""

    # 重建时临时集合带 __candidate 后缀，schema 仍应使用原集合的字段定义。
    source_collection = collection.removesuffix("__candidate")
    # 每类法律数据拥有不同主键和正文名称。
    primary_field, content_field = RECORD_FIELDS[source_collection]
    # 未明确传维度时使用 BGE-M3 的 1024 维默认值。
    dimension = dimension if dimension is not None else 1024
    # 已存在的集合不能直接假定正确，需要核对主键和向量维度。
    if client.has_collection(collection):
        # describe_collection 返回字段列表，这里转换成按字段名访问的字典。
        fields = {field["name"]: field for field in client.describe_collection(collection)["fields"]}
        # 主键错误通常表示连接到了旧库或错误集合，继续写入会破坏检索逻辑。
        if not fields.get(primary_field, {}).get("is_primary"):
            raise ValueError(f"{collection} 主键应为 {primary_field}，请检查数据库连接。")
        # 读取已有 embedding 字段的实际维度，缺失时用 -1 触发明确报错。
        actual_dimension = int(fields.get("embedding", {}).get("params", {}).get("dim", -1))
        # 查询向量与库中向量维度必须完全相同，否则 Milvus 无法计算相似度。
        if actual_dimension != dimension:
            raise ValueError(f"{collection} 向量维度不一致：应为 {dimension}，实际为 {actual_dimension}")
        # 结构符合要求，无需重复创建。
        return

    # 只有确实需要建集合时才导入 pymilvus，降低普通文件处理的依赖负担。
    from pymilvus import DataType, MilvusClient

    # auto_id=False 表示主键由我们的数据提供；动态字段用于兼容各集合额外业务字段。
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    # 法律记录主键采用字符串，最大 128 字符。
    schema.add_field(primary_field, DataType.VARCHAR, is_primary=True, max_length=128)
    # embedding 保存 BGE-M3 生成的浮点向量。
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
    # 正文字段用于生成答案和显示引用，设置较大长度上限。
    schema.add_field(content_field, DataType.VARCHAR, max_length=65535)
    # 创建索引参数对象。
    indexes = client.prepare_index_params()
    # COSINE 比较向量方向相似度；AUTOINDEX 让 Milvus 自动选择合适实现。
    indexes.add_index("embedding", metric_type="COSINE", index_type="AUTOINDEX")
    # 按已经明确的 schema 和索引创建集合；失败时让异常直接向上传递。
    client.create_collection(collection, schema=schema, index_params=indexes)


def _milvus_client(settings=None, client=None):
    """取得 Milvus 客户端；测试对象优先，正式运行按配置连接。"""

    # 测试或上层已经创建客户端时直接复用，避免额外连接。
    if client is not None:
        return client
    # 延迟导入使只做 JSON 预览的程序不必加载 Milvus SDK。
    from pymilvus import MilvusClient

    # 没传 settings 时从项目配置读取地址、令牌和数据库名。
    config = settings or get_settings()
    # db_name 明确选择持久化知识库所在数据库，避免误连默认库。
    return MilvusClient(uri=config.milvus_uri, token=config.milvus_token, db_name=config.milvus_database)


def _insert_public_batches(client, collection: str, rows: list[dict], batch_size: int = 100) -> int:
    """分批写入、刷新，然后返回数据库中的实际记录数。"""

    # range 每次前进 batch_size，从 0、100、200……依次取批次起点。
    for start in range(0, len(rows), batch_size):
        # 切片只取本批记录，防止一次请求过大。
        client.insert(collection, rows[start:start + batch_size])
    # flush 要求 Milvus 把暂存数据落盘，使随后统计能看到完整结果。
    client.flush(collection)
    # 读取集合统计并把 row_count 统一转成整数；没有统计时按 0 处理。
    return int((client.get_collection_stats(collection) or {}).get("row_count", 0))


def _drop_if_exists(client, collection: str) -> None:
    """仅在集合存在时删除，避免“集合不存在”异常打断重建清理。"""

    # 先判断再删除，调用者无需重复编写这一保护条件。
    if client.has_collection(collection):
        client.drop_collection(collection)


def rebuild_public_collections(*, settings=None, collections: dict[str, list[dict]], model=None, client=None, embedding_dimension: int | None = None):
    """使用候选集合安全重建全部公共知识库集合。

    每个正式集合先在 ``集合名__candidate`` 中生成并核对数量，成功后才替换旧集合。
    这样能尽量避免“旧集合先删了，新数据却写入失败”的情况。
    """

    # 第一步先检查全部记录，坏数据不能触发建表、删表或付费模型调用。
    validate_public_input(collections)
    # 有数据却没有嵌入模型时无法产生可检索向量，因此立即报错。
    if model is None and any(collections.values()):
        raise ValueError("公共入库需要提供嵌入模型，不能写入空向量。")
    # 调用者未指定维度时读取项目配置，通常 BGE-M3 为 1024 维。
    if embedding_dimension is None:
        embedding_dimension = (settings or get_settings()).embedding_dim
    # 获取真实或测试用 Milvus 客户端。
    target = _milvus_client(settings, client)
    # result 最终保存每个正式集合的数据库记录数。
    result = {}
    # 每个集合单独完成“向量化 -> 候选写入 -> 数量核对 -> 替换”。
    for collection, rows in collections.items():
        # 候选集合与正式集合分开，写入失败时不会直接污染正式集合。
        candidate = f"{collection}__candidate"
        # 先生成、校验向量和字段，再进行任何删除或建集合操作。
        payload = _public_rows(rows, model, embedding_dimension, collection)
        # try 确保中途失败时可以清理残留候选集合。
        try:
            # 删除上次失败可能遗留的同名候选集合。
            _drop_if_exists(target, candidate)
            # 按正式集合的字段定义创建新的候选集合。
            _ensure_public_collection(target, candidate, embedding_dimension)
            # 分批写入候选集合，并取得数据库实际记录数。
            count = _insert_public_batches(target, candidate, payload)
            # 数量不一致说明存在漏写，不能用不完整候选集合替换正式集合。
            if count != len(rows):
                raise RuntimeError(f"{collection} 候选集合行数不一致：应为 {len(rows)}，实际为 {count}")
            # 候选集合验证成功后才删除原正式集合。
            _drop_if_exists(target, collection)
            # 将候选集合改名为正式集合名，完成切换。
            try:
                target.rename_collection(candidate, collection)
            except Exception:
                # 某些兼容环境可能留下同名集合；再次确认删除后重试改名。
                _drop_if_exists(target, collection)
                target.rename_collection(candidate, collection)
            # 记录这个集合最终写入的行数。
            result[collection] = count
        except Exception:
            # 任一步失败都删除候选集合，避免下次运行读到半成品。
            _drop_if_exists(target, candidate)
            # 原异常继续交给上层显示，不能把入库失败伪装成成功。
            raise
    # 返回“集合名 -> 记录数”的重建结果。
    return result


def append_public_collections(*, settings=None, collections: dict[str, list[dict]], model=None, client=None, embedding_dimension: int | None = None):
    """把新批次公共记录追加到现有集合，不主动删除旧数据。"""

    # 追加同样必须先检查全批数据，防止一部分集合写入后才发现错误。
    validate_public_input(collections)
    # 公共检索依赖向量，有数据时禁止缺少嵌入模型。
    if model is None and any(collections.values()):
        raise ValueError("公共入库需要提供嵌入模型，不能写入空向量。")
    # 未传维度时使用项目统一配置。
    if embedding_dimension is None:
        embedding_dimension = (settings or get_settings()).embedding_dim
    # 建立或复用 Milvus 连接。
    target = _milvus_client(settings, client)
    # result 用来汇总每个集合追加后的总行数。
    result = {}
    # 逐集合生成向量并写入；调用者应保证没有重复主键。
    for collection, rows in collections.items():
        # 生成符合 Milvus schema 的白名单字段和向量。
        payload = _public_rows(rows, model, embedding_dimension, collection)
        # 集合不存在就创建，存在则核对主键与向量维度。
        _ensure_public_collection(target, collection, embedding_dimension)
        # 分批写入并保存数据库返回的实际总行数。
        result[collection] = _insert_public_batches(target, collection, payload)
    # 返回全部集合的入库统计。
    return result


# 这三个别名兼容旧上传流程的状态回调名称；当前都复用统一用户文档处理函数。
mark_ready = process_user_document
mark_failed = process_user_document
mark_processing = process_user_document


# __all__ 明确规定其他模块使用 ``from data_pipeline.index import *`` 时可见的公共名称。
__all__ = [
    "PUBLIC_COLLECTIONS", "append_public_collections", "build_public", "build_public_records",
    "file_metadata", "index", "metadata", "ocr_document", "process", "process_user_document",
    "process_user_file", "process_public_file", "rebuild_public_collections", "save", "save_processed_result", "save_public_records", "validate_public_input",
]
