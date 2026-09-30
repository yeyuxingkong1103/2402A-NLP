# -*- coding: utf-8 -*-
"""上传的校验与落盘 —— 从 ``api/app.py`` 拆出。

设计取舍：**不接收整个 ``app_state``，只收 ``documents`` 与 ``config``**
---------------------------------------------------------------
原实现 ``_sync_prepare_uploads(app_state, files)`` 只用到 ``app_state`` 的两个属性
（``documents`` 与 ``config``）。让服务层依赖整个 ``AppState`` 会把"HTTP 应用状态"
与"上传业务"绑在一起 —— 想单独测上传就得先造一个 AppState。
现在改为显式传参，服务层只依赖它真正需要的东西。

这是本项目拆分层时的通用原则：**服务层不认 ``request`` / ``app_state``，
只认自己要用的值**；线程池卸载（``run_blocking``）留在路由层做。
"""
from __future__ import annotations

from typing import Any

from ... import metrics as M
from ...logging_setup import get_logger
from ..constants import REJECT_COUNTER, SNIFF_LIMIT
from ..uploads import UploadRejected, validate_upload

__all__ = ["prepare_uploads_sync", "sample_text"]

logger = get_logger("legal_rag.api.services.uploads")


def sample_text(path: Any) -> tuple[str, str]:
    """读一篇已落盘资料的全部文本（入库前判定用）。返回 ``(文本, 备注)``。

    **抽不出文本要如实上报**（返回可读备注），不静默当空 ——
    否则"判定相符"会建立在一篇根本没读到的文件上。
    """
    from ...ingest.loaders import load_document

    try:
        document = load_document(path)
        return document.text or "", ""
    except Exception as exc:  # noqa: BLE001 - 抽不出文本要如实上报，不静默当空
        logger.warning("判定取样失败：%s（%s: %s）", path, type(exc).__name__, exc)
        return "", f"{type(exc).__name__}: {exc}"


def prepare_uploads_sync(documents: Any, config: Any, files: list[Any]) -> list[dict]:
    """校验 + 落盘一批上传文件；返回给路由层用的描述 dict 列表。

    **同步函数**：读盘/写盘是阻塞操作，调用方（路由层）必须用
    ``run_blocking`` 把它卸载到线程池，否则会卡住事件循环。

    :param documents: ``DocumentService``（提供 ``uploads.save``）
    :param config: ``RagConfig``（取 ``upload.max_upload_files``）
    :param files: FastAPI ``UploadFile`` 列表
    :raises UploadRejected: 校验不通过（带 ``reason`` / ``detail``，路由层转 400）
    """
    upload_cfg = config.upload
    max_files = int(getattr(upload_cfg, "max_upload_files", 10) or 10)
    reject = M.counter(*REJECT_COUNTER)

    if not files:
        reject.inc(reason="no_files")
        raise UploadRejected("no_files", "至少上传一个文件")
    if len(files) > max_files:
        reject.inc(reason="too_many_files")
        raise UploadRejected("too_many_files",
                             f"一次最多 {max_files} 个文件，收到 {len(files)} 个")

    saved: list[dict] = []
    for upload in files:
        raw_name = upload.filename or ""
        data = upload.file.read(SNIFF_LIMIT + 1) if upload.file else b""
        # 超过上限的内容不整块读进内存：先标记为超限再走统一拒绝路径
        if data is None:
            data = b""
        try:
            safe_name, suffix = validate_upload(filename=raw_name,
                                                content_type=upload.content_type,
                                                data=data, config=config)
            path, digest, created = documents.uploads.save(data, suffix)
        except UploadRejected as exc:
            reject.inc(reason=exc.reason)
            logger.warning("上传被拒：file=%r reason=%s detail=%s",
                           raw_name, exc.reason, exc.detail)
            raise
        M.counter("upload_files_total", "成功上传的文件数", "count").inc(suffix=suffix)
        M.counter("upload_bytes_total", "成功上传的字节数（累计）", "bytes").inc(
            len(data), suffix=suffix)
        saved.append({
            "filename": safe_name,
            "stored_path": path,
            "doc_id": digest,
            "md5": digest,
            "size": len(data),
            "created": created,
        })
        logger.info("上传受理：%r -> %s（%d 字节，%s）", raw_name, path.name, len(data),
                    "新建" if created else "幂等命中")
    return saved
