"""CLI 的装配层：把命令行参数 + `.env` 拼成检索需要的东西。

与 `retrieve_search.py`（参数解析与分派）分开，因为这两件事的失败时机不同：
参数解析的错误是**使用者打错了字**，装配的错误是**环境没配好**。
把它们混在一个文件里，排查时要先分辨是哪一类。

---

## 为什么 CLI 每次重建索引，不复用服务端的常驻单例

两者是**不同进程**。65 chunks 的构建在毫秒级，重建的代价远小于"CLI 用了一份
不知道什么时候建的索引"这个不确定性 —— 而 CLI 存在的意义正是给出可信的答案
用于标定阈值。
"""

from __future__ import annotations

import sys

from . import (
    DEFAULT_ADMIT_RANK,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CANDIDATES,
    DEFAULT_COLLECTION,
    DEFAULT_LEXICAL_MIN_COVERAGE,
    DEFAULT_RRF_K,
    DEFAULT_URI,
)
from . import bundle as bundle_module


def config_value(name: str):
    """从启动期配置取一项。**取不到返回 `None`，不抛异常。**

    不抛的理由：CLI 的配置来源是 `.env`，而 `selfcheck` 根本不需要它。
    把它做成"缺一项就整体失败"会让 `selfcheck` 也依赖 `.env` 存在 ——
    那与它"不连库、不加载权重"的定位矛盾。

    真正的必需性由 `backend/api/config.py` 的 `REQUIRED_NOW` 在**服务端启动期**
    把关。CLI 只取默认值兜底，并在输出里如实说明实际用了什么。
    """

    try:
        from backend.api.config import load
    except Exception:  # noqa: BLE001 —— 配置模块导入失败不该拖垮 selfcheck
        return None
    try:
        config = load()
    except Exception:  # noqa: BLE001 —— ConfigError 等，见上
        return None
    return getattr(config, name, None)


def uri() -> str:
    return config_value("milvus_uri") or DEFAULT_URI


def collection() -> str:
    return config_value("milvus_collection") or DEFAULT_COLLECTION


def build_index(args) -> bundle_module.IndexBundle:
    """按启动期配置（可被命令行覆盖）构建索引。

    `args` 上的 `candidates` / `rrf_k` / `admit_rank` 为 `None` 时表示"不覆盖"。
    这里 MUST NOT 用 `or` 直接兜底 —— `rrf_k=0` 是个（非法的）显式输入，
    用 `or` 会把它悄悄变成默认值，而非法输入应当被报错而不是被纠正。
    """

    from . import lexical

    lexical.warm_up()

    return bundle_module.build_index(
        uri=uri(),
        collection=collection(),
        candidates=_pick(args, "candidates", "retrieval_candidates", DEFAULT_CANDIDATES),
        rrf_k=_pick(args, "rrf_k", "rrf_k", DEFAULT_RRF_K),
        admit_rank=_pick(args, "admit_rank", "lexical_admit_rank", DEFAULT_ADMIT_RANK),
        min_coverage=_pick(args, "min_coverage", "lexical_min_coverage",
                           DEFAULT_LEXICAL_MIN_COVERAGE),
        bm25_k1=config_value("bm25_k1") or DEFAULT_BM25_K1,
        bm25_b=config_value("bm25_b") or DEFAULT_BM25_B,
    )


def _pick(args, arg_name: str, config_name: str, fallback):
    """命令行 > `.env` > 常量默认值。"""

    value = getattr(args, arg_name, None)
    if value is not None:
        return value
    return config_value(config_name) or fallback


def encode(question: str) -> list[float]:
    """算查询向量。**复用 S8 的编码器，MUST NOT 另起一套编码。**

    这是 CLI 与运行时唯一的分歧点：服务端的向量来自 `POST /ask` 时 S8 算好的
    那一份，CLI 没有请求上下文，只能现算。但两者**用的是同一个
    `backend/embed/model.py`** —— 口径仍然只有一处（specs/004 的 D3 裁决）。
    """

    from backend.embed import MAX_LENGTH, MODEL_DIR
    from backend.query.service import get_encoder

    print(
        "正在加载 BGE-M3 权重（约 2.3 GB，首次约 10 秒）…",
        file=sys.stderr,
        flush=True,
    )
    values = get_encoder().encode_query(question).tolist()

    print(
        "查询向量: %d 维（模型 %s，max_length=%d）" % (len(values), MODEL_DIR, MAX_LENGTH),
        file=sys.stderr,
        flush=True,
    )
    return values


def source_note(index: bundle_module.IndexBundle, args) -> str:
    """`search` 输出里那行"语料: …"。**必须如实说明一致性** ——

    一份被截断或与清单不符的语料看起来完全正常（检索照样返回结果，只是少了些），
    这行字是唯一能让使用者察觉的地方。
    """

    matched = index.matches_manifest
    if matched is None:
        consistency = "清单未提供，无法比对"
    elif matched:
        consistency = "与清单一致"
    else:
        consistency = "**与清单不一致**（库被改过？见 corpus 子命令）"

    parts = ["%d chunks" % index.size, consistency, "collection=%s" % index.collection]
    if getattr(args, "no_semantic", False):
        parts.append("**已跳过语义路**（标定模式）")
    if getattr(args, "no_lexical", False):
        parts.append("**已跳过关键词路**（标定模式）")
    return "，".join(parts)
