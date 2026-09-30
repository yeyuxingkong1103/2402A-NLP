"""跨模块共享的环境规避。

为什么单独成一个文件：同一个环境缺陷有两条独立的踩中路径——精排的
`sentence_transformers` 与编码的 `FlagEmbedding`，两者内部都是"先 torch 后
sklearn"的顺序。规避动作只在这里定义一份、两条导入链共用；将来环境修好时
也只删这一处，不会出现"删了精排的、漏了编码的"——留半份规避比完全没有更
危险，因为没人知道哪条路还在靠它垫背。

本模块是**环境规避、不是业务依赖**：删掉它不会改变任何业务语义，只是本机
导入会崩。实测记录与删除条件见 `ensure_import_order`。
"""
from __future__ import annotations


def ensure_import_order():
    """把 pandas 提前拉进 sys.modules，绕开 torch→sklearn 的 Windows 堆损坏。

    精确条件（2026-09-27 控制者用对照矩阵独立复现）：**pandas 必须先于
    sklearn（含 sentence_transformers / FlagEmbedding 的整条导入链）进入
    sys.modules**；相对 torch 的先后无所谓。对照矩阵（每格 3/3 一致）：

    * `torch` → `sklearn`：崩（Windows 堆损坏 0xc0000374）
    * `torch` → `pandas` → `sklearn`：正常
    * `pandas` → `torch` → `sklearn`：正常
    * `numpy` → `torch` → `sklearn`：崩
    * 裸 `import FlagEmbedding`：段错误（exit 139）
    * `pandas` → `import FlagEmbedding`：正常

    删除条件（照这份清单执行才算干净；缺 ②③④ 任一步会得到
    ModuleNotFoundError: app.compat，而不是"确认不崩"的结论）：torch / sklearn
    升级后（或 CRT/DLL 冲突修复后）依次删 ① 本函数 ensure_import_order 与
    本文件 backend/app/compat.py；② backend/tests/test_compat.py（含调用点
    源码顺序断言，调用点一删它必红，不先删它复跑必红）；③ retrieval/rerank.py
    与 ingest/embed.py 里 `from app.compat import ensure_import_order` 两行，
    以及各自 load_* 函数体内的调用两行；④ requirements.txt 的 `pandas==3.0.5`
    行。最后复跑 test_embed.py、test_rerank.py 确认不崩（test_compat.py 已删，
    不在此列）。堆损坏属 UB，"不崩"不等于"绝对安全"，所以不能把它当长期
    护身符留在生产路径上。
    """
    # 只为触发导入顺序，本函数不碰 pandas 的任何接口
    import pandas  # noqa: F401
