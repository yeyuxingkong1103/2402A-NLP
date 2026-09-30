# -*- coding: utf-8 -*-
"""测试夹具：让测试进程不与应用进程争抢 Qdrant 嵌入式文件锁。

背景
----
本项目用的是 Qdrant 嵌入式模式（`QdrantClient(path=...)`），它对存储目录持有
**独占文件锁**，同一时刻只允许一个进程打开（见 qdrant_store.py 与 README §3.4）。
因此当后端（uvicorn，监听 8000）正在运行时，测试进程直接打开
`<repo>/qdrant_storage` 会抛：

    RuntimeError: Storage folder ... is already accessed by another instance
                  of Qdrant client.

而本任务的语境是「后端保持运行、测试并行执行」，二者不可兼得，除非测试自带隔离。

策略
----
会话级夹具 `isolated_qdrant`，**显式请求才生效**（不再是 autouse）：

1. 后端**未**运行 → 直接打开真实目录（零拷贝，快）。
2. 后端**在**运行 → 把 `qdrant_storage` 快照到**固定路径**的临时目录（跳过 `.lock`
   占位文件），把 `config.QDRANT_DIR` 重定向到快照，再走正常的 `get_client()` 路径。

两种情况下测试都只读快照/真实目录、从不写生产数据，也不打断后端。

为什么不再 `autouse`（review Minor 1）
-------------------------------------
`autouse=True` 会对 `backend/tests/` 下**每一个**测试生效，且把 `config.QDRANT_DIR`
改成快照路径是**进程级**副作用。将来任何**会写** Qdrant 的测试，都会静默写进一个
teardown 时被 `rmtree` 掉的快照：断言看似通过，但生产路径从未见过这些数据。

现在只有**显式请求**该夹具的测试（目前是 `test_hybrid_retriever_returns_documents`）
才会拿到快照，副作用可见、可控；不碰 Qdrant 的测试（`test_persona_render.py`、
`test_ragas_compat.py`）不再被牵连。

为什么用固定路径（review Minor 3）
----------------------------------
原先用 `tempfile.mkdtemp()` 每次生成新目录。若 pytest 被强杀（SIGKILL）而非
正常/Ctrl-C 退出，teardown 的 `rmtree` 不会执行，每次运行泄漏约 176MB 到 %TEMP%。

改为固定路径后，夹具在启动时**先清扫再重建**：无论上次是否被强杀，最多只残留一份，
不会无限累积。
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 固定快照路径（放 %TEMP% 而非仓库内，避免被 git / 应用扫描到）。
# 固定名字 + 启动时先清理 => 即使被 SIGKILL 也不会累积多份。
_SNAPSHOT_ROOT = Path(tempfile.gettempdir()) / "raglora_qdrant_snapshot"


@pytest.fixture(scope="session")
def isolated_qdrant():
    """显式请求后才隔离 Qdrant 存储目录（见模块 docstring）。"""
    from app.core import config
    from app.services import qdrant_store

    # 先清扫上次残留（可能是被强杀留下的），再决定用哪种模式。
    shutil.rmtree(_SNAPSHOT_ROOT, ignore_errors=True)

    snapshot_root: Path | None = None
    try:
        # 后端未运行：独占锁空闲，直接用真实目录
        qdrant_store.get_client()
    except RuntimeError:
        # 后端在运行，目录被锁：改为在快照上检索
        snapshot = _SNAPSHOT_ROOT / "qdrant_storage"
        # `.lock` 是占位文件且被后端占用，跳过它（客户端会自行重建）
        shutil.copytree(
            config.QDRANT_DIR, snapshot,
            ignore=shutil.ignore_patterns(".lock"),
        )
        config.QDRANT_DIR = snapshot
        qdrant_store._client = None
        qdrant_store.get_client()
        snapshot_root = _SNAPSHOT_ROOT

    yield

    qdrant_store.close_client()
    if snapshot_root is not None:
        shutil.rmtree(snapshot_root, ignore_errors=True)
