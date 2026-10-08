"""Milvus 交互。**本包内唯一会写库的模块。**

三条关键设计，每条都有实测依据：

1. **建表时一并建索引**（`create_collection(schema=…, index_params=…)`）。
   `docs/04 §9.2` 字面要求「写入完成后统一建索引」，但 Milvus 上做不到：
   `delete`/`query` 要求 collection 已 load，而 load 又要求向量字段已有索引。
   索引因此建在**空集合**上 —— §9.2 的意图（索引不得看到写入的中途状态）天然满足。

2. **写入是补偿式提交，不是事务。** Milvus 2.6 的 WAL 事务不暴露给客户端，批量操作
   部分失败时不回滚。所以「先取旧 → 落盘备份 → 删 → 插 → 校验 → 不符则用备份回滚」
   是唯一可达的原子性。备份文件是回滚能力的**唯一来源**。

3. **一致性级别在建表时设为 Strong。** 否则写完立刻 `count(*)` 可能读到旧值，
   把「读到旧值」误判成「少写了」并触发一次不必要的回滚。

`pymilvus` 采用**惰性导入**：`--print-env` / `--print-config` 这类只读命令在依赖
尚未安装时也能跑，只有真正要连库的路径才会报缺依赖。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from . import (
    COLLECTION_NAME,
    CONSISTENCY,
    DEFAULT_URI,
    DIM,
    EXIT_ARGS,
    EXIT_DEP,
    EXIT_ROLLBACK,
    EXIT_VALIDATION,
    FAULT_ENV,
    FAULT_INSERT_COUNT,
    INDEX_TYPE,
    MAX_LENGTHS,
    META_FIELD,
    METRIC_TYPE,
    PRIMARY_FIELD,
    SCALAR_FIELDS,
    TOKEN_ENV,
    VECTOR_FIELD,
    IngestError,
)
from .inputs import DocInputs, build_chunk_meta

# doc_id 只允许这些字符 —— 它会被拼进 Milvus 的过滤表达式，必须挡住引号等注入字符
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# 查回旧数据的上限。显式给出而不依赖 Milvus 的默认上限（16384），
# 否则超大文档的备份会被**静默截断**，回滚就不完整了。
QUERY_LIMIT = 16384

PIPELINE_HASH_FIELD = "pipeline_config_hash"


@dataclass
class CommitResult:
    doc_id: str
    old_count: int
    deleted: int
    inserted: int
    final_count: int
    expected: int
    backup_path: str | None


# ---------------------------------------------------------------- 连接与建表


def load_pymilvus():
    """惰性导入；缺依赖时报退出码 3 并给出可复制的安装命令。"""
    try:
        import pymilvus
    except ModuleNotFoundError as exc:
        raise IngestError(
            EXIT_DEP,
            "缺少依赖 pymilvus。安装：\n"
            "  D:/zg6_Project/9/med_rag/rag/python.exe -m pip install pymilvus",
        ) from exc
    return pymilvus


def _guard(op: str, exc: BaseException) -> IngestError:
    return IngestError(EXIT_DEP, "Milvus 操作失败（%s）：%s" % (op, exc))


def _token() -> str | None:
    """凭据只从环境变量读（宪法原则 III）。错误信息里不回显取值，只报变量名。"""
    value = os.environ.get(TOKEN_ENV)
    return value or None


def connect(uri: str):
    pm = load_pymilvus()
    try:
        client = pm.MilvusClient(uri=uri, token=_token())
    except Exception as exc:  # noqa: BLE001 —— 连接层失败一律归为「外部依赖不可用」
        raise _guard("连接 %s" % uri, exc) from exc
    return client


def _build_schema(pm):
    varchar = lambda name: pm.FieldSchema(  # noqa: E731
        name=name, dtype=pm.DataType.VARCHAR, max_length=MAX_LENGTHS[name]
    )
    fields = [
        pm.FieldSchema(
            name=PRIMARY_FIELD,
            dtype=pm.DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=MAX_LENGTHS[PRIMARY_FIELD],
        ),
        pm.FieldSchema(name=VECTOR_FIELD, dtype=pm.DataType.FLOAT_VECTOR, dim=DIM),
        varchar("text"),
        varchar("doc_id"),
        varchar("file_name"),
        pm.FieldSchema(name="page_start", dtype=pm.DataType.INT32),
        pm.FieldSchema(name="page_end", dtype=pm.DataType.INT32),
        varchar("section"),
        varchar("block_type"),
        varchar("source_hash"),
        varchar(PIPELINE_HASH_FIELD),
        # S4 记录里没有单独成列的那部分（text_for_embedding / heading_path / block_ids /
        # sub_index / char_len / chunk_rule_version …）。JSON 字段没有 max_length，
        # 上限由 Milvus 固定为 65536 字节，写库前由 inputs 的 V12 把关。
        pm.FieldSchema(name=META_FIELD, dtype=pm.DataType.JSON),
    ]
    # enable_dynamic_field=False：未声明的字段**报错**而不是被静默收进动态字段。
    # schema 是 docs/04 §9.1 的契约，多一个字段必须是显式决定。
    return pm.CollectionSchema(
        fields=fields,
        description="med_rag V1 —— 展示原文 + 引用定位字段（docs/04 §9.1）",
        enable_dynamic_field=False,
    )


def ensure_collection(client) -> tuple[bool, bool]:
    """把 collection 弄成所需状态。返回 (是否新建, 是否补建了索引)。

    不存在 → 建表 + 建索引 + load。
    存在 → 核对 schema（不符即报错，**不自动改 schema**）；索引缺失则补建。
    """
    pm = load_pymilvus()
    try:
        exists = client.has_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("检查 collection 是否存在", exc) from exc

    if not exists:
        try:
            # pymilvus 2.6 的 create_collection 要求 IndexParams 对象，不接受 list[dict]
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=VECTOR_FIELD,
                index_type=INDEX_TYPE,
                metric_type=METRIC_TYPE,
            )
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=_build_schema(pm),
                index_params=index_params,
                consistency_level=CONSISTENCY,
            )
        except Exception as exc:  # noqa: BLE001
            raise _guard("创建 collection（schema + FLAT/COSINE 索引）", exc) from exc
        return True, False

    assert_collection_compatible(client)
    return False, ensure_index(client)


def ensure_index(client) -> bool:
    """索引缺失时补建 —— 唯一确定，非破坏性（V1 锁定的 FLAT/COSINE）。

    这个分支是必需的：`create_collection` 是「先建表、再建索引」两步，中途任何失败
    都会留下一个没有索引的空 collection，而它既不能 load 也不能 delete/query，
    下一次运行必须能自愈，否则只能靠人工 drop 重建。
    """
    pm = load_pymilvus()
    try:
        indexes = client.list_indexes(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("读取索引列表", exc) from exc
    if indexes:
        return False

    try:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type=INDEX_TYPE,
            metric_type=METRIC_TYPE,
        )
        client.create_index(COLLECTION_NAME, index_params)
    except Exception as exc:  # noqa: BLE001
        raise _guard("补建索引（%s/%s）" % (INDEX_TYPE, METRIC_TYPE), exc) from exc
    return True


def assert_collection_compatible(client) -> None:
    """已存在时核对 schema 与**已有索引**的类型；不符即报错，**不自动改 schema**（E11）。

    索引**缺失**不在这里报错 —— 那是 `ensure_index` 的职责（补建唯一确定的索引）。
    这里只在索引已存在时校验它是不是 V1 锁定的 FLAT/COSINE。
    """
    try:
        desc = client.describe_collection(COLLECTION_NAME)
        indexes = client.list_indexes(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("读取 collection 描述", exc) from exc

    fields = {f.get("name"): f for f in (desc.get("fields") or [])}
    expected = set(SCALAR_FIELDS) | {VECTOR_FIELD}
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise IngestError(
            EXIT_VALIDATION,
            "collection %s 的字段与 docs/04 §9.1 不符：缺少 %s，多出 %s\n"
            "（不自动改 schema —— 改 schema 是破坏性动作，必须由人显式发起）\n"
            "  重建命令（65 行，源产物都在，幂等）：\n"
            "    D:/zg6_Project/9/med_rag/rag/python.exe -c \"from pymilvus import MilvusClient;"
            " MilvusClient('%s').drop_collection('%s')\"\n"
            "  然后重跑入库命令即可。" % (COLLECTION_NAME, missing or "无", extra or "无",
                                        DEFAULT_URI, COLLECTION_NAME),
        )

    vector_params = fields[VECTOR_FIELD].get("params") or {}
    if int(vector_params.get("dim", -1)) != DIM:
        raise IngestError(
            EXIT_VALIDATION,
            "collection %s 的向量维度是 %s，应为 %d" % (COLLECTION_NAME, vector_params.get("dim"), DIM),
        )

    if not indexes:
        return  # 缺失由 ensure_index 补建（见其 docstring）

    try:
        index_desc = client.describe_index(COLLECTION_NAME, indexes[0])
    except Exception as exc:  # noqa: BLE001
        raise _guard("读取索引描述", exc) from exc

    actual_type = str(index_desc.get("index_type", "")).upper()
    actual_metric = str(index_desc.get("metric_type", "")).upper()
    if actual_type != INDEX_TYPE or actual_metric != METRIC_TYPE:
        raise IngestError(
            EXIT_VALIDATION,
            "collection %s 的索引是 %s/%s，应为 %s/%s（docs/04 §9.1 的 V1 锁定值）\n"
            "MUST NOT 自动降级为别的索引类型" % (COLLECTION_NAME, actual_type, actual_metric, INDEX_TYPE, METRIC_TYPE),
        )


def ensure_loaded(client) -> None:
    try:
        state = client.get_load_state(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("读取 load 状态", exc) from exc
    text = str(state)
    if "Loaded" in text and "NotLoad" not in text:
        return
    try:
        client.load_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("load collection", exc) from exc


# ---------------------------------------------------------------- 读


def _safe_doc_id(doc_id: str) -> str:
    if not _DOC_ID_RE.match(doc_id):
        raise IngestError(EXIT_ARGS, "doc_id 含非法字符（只允许字母数字与 _ . -）：%r" % doc_id)
    return doc_id


def _filter_for(doc_id: str) -> str:
    return '%s == "%s"' % ("doc_id", _safe_doc_id(doc_id))


def count_all(client) -> int:
    try:
        res = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
    except Exception as exc:  # noqa: BLE001
        raise _guard("统计总行数", exc) from exc
    return int(res[0]["count(*)"])


def count_for_doc(client, doc_id: str) -> int:
    try:
        res = client.query(COLLECTION_NAME, filter=_filter_for(doc_id), output_fields=["count(*)"])
    except Exception as exc:  # noqa: BLE001
        raise _guard("统计 doc_id=%s 的行数" % doc_id, exc) from exc
    return int(res[0]["count(*)"])


def read_stored_hash(client) -> str | None:
    """版本门禁的读取路径：hash 是逐行字段，取任意一行即可。集合为空 → None。"""
    try:
        res = client.query(
            COLLECTION_NAME, filter="", output_fields=[PIPELINE_HASH_FIELD], limit=1
        )
    except Exception as exc:  # noqa: BLE001
        raise _guard("读取库内参数版本哈希", exc) from exc
    if not res:
        return None
    value = res[0].get(PIPELINE_HASH_FIELD)
    return str(value) if value else None


def capture_old_rows(client, doc_id: str) -> list[dict[str, Any]]:
    """查回该 doc_id 的全部行，**含向量** —— 否则回滚只能还原标量字段，数据是残的。"""
    fields = list(SCALAR_FIELDS) + [VECTOR_FIELD]
    try:
        rows = client.query(
            COLLECTION_NAME,
            filter=_filter_for(doc_id),
            output_fields=fields,
            limit=QUERY_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001
        raise _guard("取回 doc_id=%s 的旧数据" % doc_id, exc) from exc
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 写


def doc_summary(client, doc_id: str) -> dict[str, Any] | None:
    """给 index_manifest 用的「以库为准」的单文档摘要。库里没有该 doc_id → None。

    查回的行数达到 QUERY_LIMIT 时**报错而不是截断** —— 否则清单会静默少记。
    """
    try:
        rows = client.query(
            COLLECTION_NAME,
            filter=_filter_for(doc_id),
            output_fields=["file_name", "source_hash", "page_end"],
            limit=QUERY_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001
        raise _guard("汇总 doc_id=%s" % doc_id, exc) from exc
    if not rows:
        return None
    if len(rows) >= QUERY_LIMIT:
        raise IngestError(
            EXIT_VALIDATION,
            "doc_id=%s 的行数达到查询上限 %d，索引清单会不完整。请提高 store.QUERY_LIMIT。"
            % (doc_id, QUERY_LIMIT),
        )
    return {
        "doc_id": doc_id,
        "file_name": str(rows[0].get("file_name", "")),
        "source_hash": str(rows[0].get("source_hash", "")),
        "chunk_count": len(rows),
        "page_count": max(int(r["page_end"]) for r in rows),
    }


def build_entities(inp: DocInputs, config_hash: str) -> list[dict[str, Any]]:
    """向量取 `.npy` 的第 **row_index** 行并与其 chunk 绑定 —— 不依赖读文件顺序。"""
    entities: list[dict[str, Any]] = []
    for i, chunk in enumerate(inp.chunks):
        row_index = int(inp.rows[i]["row_index"])
        entities.append(
            {
                PRIMARY_FIELD: str(chunk["chunk_id"]),
                VECTOR_FIELD: inp.matrix[row_index].tolist(),
                "text": chunk["text"],  # 展示用原文，**不是** text_for_embedding
                "doc_id": str(chunk["doc_id"]),
                "file_name": str(chunk["file_name"]),
                "page_start": int(chunk["page_start"]),
                "page_end": int(chunk["page_end"]),
                "section": str(chunk.get("section") or ""),
                "block_type": str(chunk["block_type"]),
                "source_hash": str(chunk["source_hash"]),
                PIPELINE_HASH_FIELD: config_hash,
                # 其余 S4 字段整体收进 JSON 字段。值来自 json.load，全是原生类型；
                # 若混进 numpy 类型，pymilvus 会以一条**指向错误**的信息报错（issue #2886）。
                META_FIELD: build_chunk_meta(chunk),
            }
        )
    return entities


def _insert(client, entities: list[dict[str, Any]]) -> int:
    try:
        res = client.insert(COLLECTION_NAME, entities)
    except Exception as exc:  # noqa: BLE001
        raise _guard("插入 %d 行" % len(entities), exc) from exc
    count = int(res.get("insert_count", res.get("insert_cnt", 0)))
    if os.environ.get(FAULT_ENV) == FAULT_INSERT_COUNT:
        # 故障注入：只用于 quickstart §7 验证回滚路径。正常路径下该分支不进入。
        return count - 1
    return count


def _flush(client) -> None:
    try:
        client.flush(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _guard("flush", exc) from exc


def _delete_for_doc(client, doc_id: str) -> int:
    """只按 doc_id 过滤（D1）。MUST NOT 出现无过滤条件的 delete/drop。"""
    try:
        res = client.delete(COLLECTION_NAME, filter=_filter_for(doc_id))
    except Exception as exc:  # noqa: BLE001
        raise _guard("删除 doc_id=%s 的旧数据" % doc_id, exc) from exc
    return int(res.get("delete_count", res.get("delete_cnt", 0)))


@dataclass
class Backup:
    path: str | None
    rows: list[dict[str, Any]]


def save_backup(rows: list[dict[str, Any]], doc_id: str, backup_dir: str) -> str:
    """落盘备份。这是回滚失败时**人工恢复的唯一依据**，所以先写盘再删除。"""
    os.makedirs(backup_dir, exist_ok=True)
    path = os.path.join(backup_dir, "%s.rollback.jsonl" % doc_id)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    except OSError as exc:
        raise IngestError(
            EXIT_VALIDATION,
            "备份旧数据失败，为安全起见**不执行删除**：%s\n  %s" % (path, exc),
        ) from exc
    return path


def load_backup(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(EXIT_ROLLBACK, "读取备份文件失败：%s\n  %s" % (path, exc)) from exc


def _restore(client, doc_id: str, rows: list[dict[str, Any]]) -> None:
    _delete_for_doc(client, doc_id)
    if rows:
        _insert(client, rows)
        _flush(client)


def commit_document(
    client,
    inp: DocInputs,
    config_hash: str,
    backup_dir: str,
) -> CommitResult:
    """单份文档的原子提交（补偿式）。失败一律抛 IngestError，绝不静默。"""
    doc_id = inp.doc_id
    expected = inp.count

    old_count = count_for_doc(client, doc_id)
    backup = Backup(path=None, rows=[])
    if old_count:
        backup.rows = capture_old_rows(client, doc_id)
        backup.path = save_backup(backup.rows, doc_id, backup_dir)

    deleted = _delete_for_doc(client, doc_id)

    try:
        entities = build_entities(inp, config_hash)
        inserted = _insert(client, entities)
        _flush(client)
        final = count_for_doc(client, doc_id)
    except BaseException as exc:  # noqa: BLE001
        # 这里是**必须**用宽 except 的少数场合：删除已经发生，无论后续因为什么失败
        # （插入报错、flush 报错、查询报错、甚至 KeyboardInterrupt），
        # 库都停在「旧数据已删、新数据未进」的状态，必须回滚。且 _rollback 一定会抛，
        # 不会把异常吞掉。
        _rollback(client, doc_id, backup, old_count, cause=exc)

    if inserted != expected or final != expected:
        _rollback(
            client,
            doc_id,
            backup,
            old_count,
            cause=IngestError(
                EXIT_VALIDATION,
                "写入后校验不符：预期 %d 行，插入返回 %d，库内实查 %d" % (expected, inserted, final),
            ),
        )

    return CommitResult(
        doc_id=doc_id,
        old_count=old_count,
        deleted=deleted,
        inserted=inserted,
        final_count=final,
        expected=expected,
        backup_path=backup.path,
    )


def _rollback(client, doc_id: str, backup: Backup, old_count: int, cause: BaseException) -> None:
    """用备份把库恢复到运行前状态。**本函数一定抛异常**（调用方无需 return）。"""
    reason = getattr(cause, "message", None) or str(cause)
    try:
        _restore(client, doc_id, backup.rows)
        restored = count_for_doc(client, doc_id)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            EXIT_ROLLBACK,
            "写入失败，且**回滚也失败** —— 库状态不确定，需人工介入。\n"
            "  失败原因：%s\n"
            "  回滚失败：%s\n"
            "  人工恢复依据：%s"
            % (reason, exc, backup.path or "（本次运行前库内无该文档的数据，直接删除 doc_id 即可）"),
        ) from exc

    if restored != old_count:
        raise IngestError(
            EXIT_ROLLBACK,
            "写入失败，回滚后行数仍不符（预期 %d，实际 %d）—— 库状态不确定，需人工介入。\n"
            "  失败原因：%s\n"
            "  人工恢复依据：%s" % (old_count, restored, reason, backup.path or "（无备份）"),
        ) from cause

    raise IngestError(
        EXIT_VALIDATION,
        "写入失败，已回滚到运行前状态（该文档 %d 行）。\n  原因：%s" % (old_count, reason),
    ) from cause


__all__ = [
    "Backup",
    "CommitResult",
    "assert_collection_compatible",
    "build_entities",
    "capture_old_rows",
    "commit_document",
    "connect",
    "count_all",
    "count_for_doc",
    "ensure_collection",
    "ensure_index",
    "ensure_loaded",
    "load_backup",
    "load_pymilvus",
    "read_stored_hash",
    "save_backup",
]
