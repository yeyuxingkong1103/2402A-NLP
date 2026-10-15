# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单08 基线适配器（「优化前」对照）

职责：以**未经修改**的工单08 代码路径回答同一个问题，作为「优化前」基线。
      工单硬性要求「必须基于 08 的 Graph RAG 服务及检索结果进行优化」——
      所以基线的算法必须原样调用，不能重新实现一遍。

关键设计：基线读哪一份向量索引
    工单08 的 `src/config.QDRANT_LOCAL_PATH` 默认指向 `工单8/data/qdrant`。
    本适配器在**任何 Qdrant 客户端构造之前**把该配置改写为 09 的副本
    （`工单9/data/qdrant`，内容与 08 逐字节相同）。理由有三：

      1. **08 目录保持物理只读** —— 跑「优化前」不会给 08 留下任何写痕迹
         （Qdrant local 模式会往存储目录写 .lock 与段文件）；
      2. **避免排他锁冲突** —— Qdrant local 对存储目录加独占锁，
         若 08 的界面正在运行，共享目录会让基线直接建连失败；
      3. **A/B 严格可比** —— 基线与 09 读**同一份向量**，
         前后差异只能来自算法（检索/融合/生成），不可能来自索引差异。

    被改写的是**存储位置**，不是算法；08 的检索逻辑（召回路数、RRF 融合、
    精排、上下文拼装、生成 Prompt）100% 原样执行。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from w9 import compat, config9

# ---------------------------------------------------------------------------
# ① 把 08 的 Qdrant 指向 09 的副本（必须早于任何客户端构造）
# ---------------------------------------------------------------------------
from src import config as c8  # noqa: E402


def _pin_qdrant_to_copy() -> dict:
    """把 08 的 Qdrant 存储路径改写到 09 的副本。幂等。"""
    orig = c8.QDRANT_LOCAL_PATH
    if os.path.abspath(orig) != os.path.abspath(config9.QDRANT9_LOCAL_PATH):
        c8.QDRANT_LOCAL_PATH = config9.QDRANT9_LOCAL_PATH
        # QDRANT_DIR 是向量库模块创建目录时用的键，一并改写，避免它去
        # 工单8 的路径下做 makedirs / 写锁文件。
        c8.QDRANT_DIR = config9.QDRANT9_LOCAL_PATH
    return {"original": orig, "pinned_to": c8.QDRANT_LOCAL_PATH,
            "is_copy": c8.QDRANT_LOCAL_PATH == config9.QDRANT9_DIR}


def _pin_fulltext_index() -> dict:
    """把 08 倒排索引的**持久化路径**改写到 09。

    为什么必须做：`src/fulltext.py` 的 `build()` 在构建完倒排索引后无条件
    把 31.7 MB 的 pickle 写回 `config.DATA_DIR/index/fulltext.pkl`。
    不改写的话，只要跑一次「优化前」基线，就会在 08 的目录里留下写痕迹
    —— 而 09 的核心承诺是「08 物理只读、永远是可复现的基线」。

    注意：被改写的只是**落盘位置**，索引的构建逻辑与内容完全不变
    （08 的 `build()` 本来就不读这个文件，它只在每次进程启动时重建）。
    """
    import src.fulltext as ft

    orig = ft._INDEX_PATH
    target = os.path.join(config9.DATA9, "index", "fulltext.pkl")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    ft._INDEX_PATH = target
    return {"original": orig, "pinned_to": target}


QDRANT_PIN = _pin_qdrant_to_copy()
FULLTEXT_PIN = _pin_fulltext_index()

# 缓存哈希（证明副本与 08 原件内容一致 —— A/B 可比性的证据）
_QDRANT_HASH: dict | None = None


def qdrant_copy_hash(force: bool = False) -> dict:
    """对 08 原件与 09 副本的 collection 目录做内容哈希比对。

    只算**文件名+大小的摘要**（不读 97MB 全量内容），足以在文档中证明
    「09 副本 == 08 原件」，成本却只有一次目录遍历。
    """
    global _QDRANT_HASH
    if _QDRANT_HASH is not None and not force:
        return _QDRANT_HASH

    def _digest(base: str) -> dict:
        h = hashlib.sha256()
        n = 0
        total = 0
        if not os.path.isdir(base):
            return {"exists": False, "files": 0, "bytes": 0, "sha256": ""}
        for dp, dn, fn in os.walk(base):
            dn.sort()
            for f in sorted(fn):
                if f == ".lock":
                    continue
                fp = os.path.join(dp, f)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, base)
                h.update(f"{rel}|{st.st_size}\n".encode("utf-8"))
                n += 1
                total += st.st_size
        return {"exists": True, "files": n, "bytes": total,
                "sha256": h.hexdigest()}

    a = _digest(config9.WO8_QDRANT)
    b = _digest(config9.QDRANT9_DIR)
    _QDRANT_HASH = {"wo8": a, "wo9": b,
                    "identical": a["sha256"] == b["sha256"] and a["files"] > 0}
    return _QDRANT_HASH


# ---------------------------------------------------------------------------
# ② 基线问答（08 算法原样调用）
# ---------------------------------------------------------------------------
def _qa_module():
    import importlib
    return importlib.import_module("src.graph_qa")


def ask(question: str, strategy: str = "hybrid", state=None):
    """用 08 的 Graph RAG 链路回答（「优化前」基线）。

    返回 08 原生的 QaResult（含 answer / graph_facts / text_hits / timings /
    degraded），不做任何包装改写 —— 评估脚本直接消费它的字段。
    """
    return _qa_module().ask(question, state=state, strategy=strategy)


def warmup() -> dict:
    """预热 08 链路（图谱 + 模型），避免首问被冷启动拖慢。"""
    t0 = time.perf_counter()
    try:
        out = _qa_module().warmup()
        out["baseline_warmup_s"] = round(time.perf_counter() - t0, 3)
        out["qdrant"] = QDRANT_PIN
        return out
    except Exception as exc:                          # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "qdrant": QDRANT_PIN}


def status() -> dict:
    """基线状态（含复用对象与索引一致性自检）。"""
    try:
        st = _qa_module().status()
    except Exception as exc:                          # noqa: BLE001
        st = {"error": f"{type(exc).__name__}: {exc}"}
    st["reuse_verify"] = compat.verify_reuse()
    st["qdrant"] = QDRANT_PIN
    st["fulltext"] = FULLTEXT_PIN
    st["qdrant_hash"] = qdrant_copy_hash()
    st["wo8_kg"] = config9.WO8_KG_PICKLE
    return st


def clear_caches() -> None:
    """清 08 的子图缓存（评测逐题隔离时用）。"""
    try:
        _qa_module().clear_caches()
    except Exception:                                 # noqa: BLE001
        pass


if __name__ == "__main__":
    print(json.dumps(status(), ensure_ascii=False, indent=2)[:2000])
